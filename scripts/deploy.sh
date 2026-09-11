#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$1"
stage=$2
version=$3
metadata=$4
exec 9>.deploy.lock
flock -n 9 || { echo 'Another deployment is running' >&2; exit 1; }
previous=$(readlink runs/current)
container=$(docker compose ps -q api)
test -n "$container"
image=$(docker inspect --format '{{.Image}}' "$container")
umask 077
docker compose config > "$stage/compose.yaml"
printf 'services:\n  api:\n    image: %s\n' "$image" > "$stage/image.yaml"
cutover=0
swapped=0
had_run=0
rollback() {
    docker compose stop api || return
    if [ "$swapped" = 1 ]; then
        mv -- "runs/$version" "$stage/artifacts" || return
    fi
    if [ "$had_run" = 1 ]; then
        mv -- "$stage/previous-run" "runs/$version" || return
    fi
    for file in "$stage"/catalogs/*; do
        [ -f "$file" ] || continue
        mv -f -- "$file" "data/$(basename "$file")" || return
    done
    ln -sfn -- "$previous" runs/.current || return
    mv -Tf -- runs/.current runs/current || return
    docker compose --project-directory "$PWD" -f "$stage/compose.yaml" -f "$stage/image.yaml" up -d --no-deps --no-build --pull never --force-recreate --wait --wait-timeout 180 api
}
finish() {
    status=$?
    trap - EXIT
    if [ "$status" != 0 ] && [ "$cutover" = 1 ]; then
        docker compose logs --tail 80 api >&2 || true
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
echo 'Building API while the current container stays online'
docker compose build api
if [ "$metadata" = 1 ]; then
    mkdir "$stage/catalogs"
    for file in beatmaps.parquet beatmapsets.parquet strains.parquet; do
        ln -- "data/$file" "$stage/catalogs/$file"
    done
fi
echo 'Switching API'
cutover=1
docker compose stop api
if [ -e "runs/$version" ]; then
    mv -- "runs/$version" "$stage/previous-run"
    had_run=1
fi
mv -- "$stage/artifacts" "runs/$version"
swapped=1
if [ "$metadata" = 1 ]; then
    for file in beatmaps.parquet beatmapsets.parquet strains.parquet; do
        mv -f -- "$stage/$file" "data/$file"
    done
fi
ln -sfn -- "$version" runs/.current
mv -Tf -- runs/.current runs/current
docker compose up -d --no-deps --no-build --pull never --force-recreate --wait --wait-timeout 180 api
cutover=0
python3 -c 'import shutil, sys; shutil.rmtree(sys.argv[1])' "$stage"
