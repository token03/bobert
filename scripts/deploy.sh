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
echo 'Updating code'
git pull --ff-only
previous=$(readlink runs/current)
container=$("${compose[@]}" ps -q api)
test -n "$container"
image=$(docker inspect --format '{{.Image}}' "$container")
"${compose[@]}" config > "$stage/compose.yaml"
printf 'services:\n  api:\n    image: %s\n' "$image" > "$stage/image.yaml"
echo "Fetching $repo@$version"
mkdir -p "$stage/artifacts" "$stage/catalogs"
base="https://huggingface.co/$repo/resolve/$version"
curl -fsSL --retry 3 --retry-delay 2 -o "$stage/artifacts/model.safetensors" "$base/model.safetensors"
curl -fsSL --retry 3 --retry-delay 2 -o "$stage/artifacts/embeddings.parquet" "$base/embeddings.parquet"
curl -fsSL --retry 3 --retry-delay 2 -o "$stage/catalogs/beatmaps.parquet" "$base/data/beatmaps.parquet"
curl -fsSL --retry 3 --retry-delay 2 -o "$stage/catalogs/beatmapsets.parquet" "$base/data/beatmapsets.parquet"
curl -fsSL --retry 3 --retry-delay 2 -o "$stage/catalogs/strains.parquet" "$base/data/strains.parquet"
echo 'Building API while the current container stays online'
"${compose[@]}" build api
echo 'Switching API'
cutover=1
"${compose[@]}" stop api
if [ -e "runs/$version" ]; then
    mv -- "runs/$version" "$stage/previous-run"
    had_run=1
fi
mv -- "$stage/artifacts" "runs/$version"
swapped=1
mkdir "$stage/previous-catalogs"
had_catalogs=1
for file in beatmaps.parquet beatmapsets.parquet strains.parquet; do
    if [ -f "data/$file" ]; then
        mv -- "data/$file" "$stage/previous-catalogs/$file"
    fi
    mv -- "$stage/catalogs/$file" "data/$file"
done
ln -sfn -- "$version" runs/.current
mv -Tf -- runs/.current runs/current
"${compose[@]}" up -d --no-deps --no-build --pull never --force-recreate --wait --wait-timeout 180 api
cutover=0
python3 -c 'import shutil, sys; shutil.rmtree(sys.argv[1])' "$stage"
