#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$1"
stage=$2
version=$3
repo=$4
compose=(docker compose -f compose.yaml -f compose.prod.yaml)
exec 9>.deploy.lock
flock -n 9 || { echo 'Another deployment is running' >&2; exit 1; }
umask 077
cutover=0
swapped=0
had_run=0
had_catalogs=0
rollback() {
    "${compose[@]}" stop api || return
    if [ "$swapped" = 1 ]; then
        mv -- "runs/$version" "$stage/artifacts" || return
    fi
    if [ "$had_run" = 1 ]; then
        mv -- "$stage/previous-run" "runs/$version" || return
    fi
    if [ "$had_catalogs" = 1 ]; then
        for file in beatmaps.parquet beatmapsets.parquet strains.parquet; do
            if [ -f "data/$file" ]; then
                mv -- "data/$file" "$stage/catalogs/$file" || return
            fi
            if [ -f "$stage/previous-catalogs/$file" ]; then
                mv -- "$stage/previous-catalogs/$file" "data/$file" || return
            fi
        done
    fi
    ln -sfn -- "$previous" runs/.current || return
    mv -Tf -- runs/.current runs/current || return
    docker compose --project-directory "$PWD" -f "$stage/compose.yaml" -f "$stage/image.yaml" run --rm -T --no-deps --entrypoint /bin/sh api -c 'rm -f /app/cache/runtime*.sqlite*' || return
    docker compose --project-directory "$PWD" -f "$stage/compose.yaml" -f "$stage/image.yaml" up -d --no-deps --no-build --pull never --force-recreate --wait --wait-timeout 180 api
}
finish() {
    status=$?
    trap - EXIT
    if [ "$status" != 0 ] && [ "$cutover" = 1 ]; then
        "${compose[@]}" logs --tail 80 api >&2 || true
        if rollback; then
            echo 'Deployment failed; previous API restored' >&2
        else
            echo "Rollback failed; recovery files retained at $stage" >&2
            exit 1
        fi
    fi
    docker image prune -af || status=1
    docker builder prune -af || status=1
    if [ "$status" = 0 ]; then
        echo "Deployed $version"
    else
        echo "Deployment files retained at $stage" >&2
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
previous=$(readlink runs/current)
container=$("${compose[@]}" ps -q api)
test -n "$container"
image=$(docker inspect --format '{{.Image}}' "$container")
"${compose[@]}" config > "$stage/compose.yaml"
printf 'services:\n  api:\n    image: %s\n' "$image" > "$stage/image.yaml"
echo 'Updating code'
git pull --ff-only
echo 'Building API while the current container stays online'
"${compose[@]}" build api
echo "Fetching $repo@$version"
(
    umask 022
    mkdir -p "$stage/artifacts" "$stage/catalogs"
    "${compose[@]}" run --rm -T --no-deps --user 0 -e HF_HUB_DISABLE_PROGRESS_BARS=1 -v "$stage/artifacts:/app/release" -v "$stage/catalogs:/app/catalogs" api python - "$repo" "$version" <<'PYTHON'
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from core.artifacts import CATALOGS, RUN_MANIFEST, read_run, resolve_artifacts

os.umask(0o022)
repo, version = sys.argv[1:]
revision = HfApi(token=False).model_info(repo, revision=version).sha
release = Path("/app/release")
snapshot_download(
    repo, revision=revision, token=False, local_dir=release, allow_patterns=[RUN_MANIFEST]
)
paths = []
for mode in read_run(release):
    artifacts = resolve_artifacts(release, mode)
    paths.extend(
        str(path.relative_to(release)) for path in (artifacts.model, artifacts.index)
    )
snapshot_download(
    repo,
    revision=revision,
    token=False,
    local_dir=release,
    allow_patterns=[*paths, *(f"data/{name}" for name in CATALOGS)],
    max_workers=4,
)
for name in CATALOGS:
    shutil.move(str(release / "data" / name), str(Path("/app/catalogs") / name))
(release / "data").rmdir()
shutil.rmtree(release / ".cache", ignore_errors=True)
print(f"Downloaded {repo}@{revision}", flush=True)
PYTHON
)
echo 'Validating release artifacts'
"${compose[@]}" run --rm -T --no-deps -v "$stage/artifacts:/app/release:ro" -v "$stage/catalogs:/app/catalogs:ro" api python -c 'from core.artifacts import validate_catalogs, validate_run; validate_run("/app/release"); validate_catalogs("/app/catalogs")'
mkdir "$stage/previous-catalogs"
for file in beatmaps.parquet beatmapsets.parquet strains.parquet; do
    if [ -f "data/$file" ]; then
        cp -p -- "data/$file" "$stage/previous-catalogs/$file"
    fi
done
echo 'Switching API'
cutover=1
"${compose[@]}" stop api
if [ -e "runs/$version" ]; then
    mv -- "runs/$version" "$stage/previous-run"
    had_run=1
fi
mv -- "$stage/artifacts" "runs/$version"
swapped=1
had_catalogs=1
for file in beatmaps.parquet beatmapsets.parquet strains.parquet; do
    mv -- "$stage/catalogs/$file" "data/$file"
done
ln -sfn -- "$version" runs/.current
mv -Tf -- runs/.current runs/current
"${compose[@]}" run --rm -T --no-deps --entrypoint /bin/sh api -c 'rm -f /app/cache/runtime*.sqlite*'
"${compose[@]}" up -d --no-deps --no-build --pull never --force-recreate --wait --wait-timeout 180 api
cutover=0
python3 -c 'import shutil, sys; shutil.rmtree(sys.argv[1])' "$stage"
