#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$1"
stage=$2
version=$3
repo=$4
compose=(docker compose -f compose.yaml -f compose.prod.yaml)
catalogs=(beatmaps.parquet beatmapsets.parquet strains.parquet)
exec 9>.deploy.lock
flock -n 9 || { echo 'Another deployment is running' >&2; exit 1; }
cutover=0
previous=$(readlink runs/current)
if [ -e "runs/$version" ]; then
    echo "runs/$version already exists" >&2
    exit 1
fi
container=$("${compose[@]}" ps -q api)
test -n "$container"
image=$(docker inspect --format '{{.Image}}' "$container")
"${compose[@]}" config > "$stage/compose.yaml"
printf 'services:\n  api:\n    image: %s\n' "$image" > "$stage/image.yaml"
rollback() {
    for file in "${catalogs[@]}"; do
        mv -f -- "$stage/previous/$file" "data/$file" || return
    done
    ln -sfn -- "$previous" runs/.current && mv -Tf -- runs/.current runs/current || return
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
        rm -rf -- "$stage"
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
echo 'Building images while the current containers stay online'
"${compose[@]}" build
echo "Fetching changes in $repo@$version"
mkdir -p "$stage/release/data" "$stage/previous"
cp -al -- "runs/$previous/." "$stage/release/"
for file in "${catalogs[@]}"; do
    cp -al -- "data/$file" "$stage/release/data/$file"
    cp -al -- "data/$file" "$stage/previous/$file"
done
"${compose[@]}" run --rm -T --no-deps --user 0 \
    -e HF_TOKEN="$(sed -n 's/^HF_TOKEN=//p' .env)" -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -v "$stage/release:/app/release" \
    api python -m server.release "$repo" "$version" /app/release
echo 'Switching API'
cutover=1
for file in "${catalogs[@]}"; do
    mv -f -- "$stage/release/data/$file" "data/$file"
done
rmdir -- "$stage/release/data"
mv -- "$stage/release" "runs/$version"
ln -sfn -- "$version" runs/.current
mv -Tf -- runs/.current runs/current
"${compose[@]}" up -d --no-build --pull never --force-recreate --wait --wait-timeout 180 api bot
cutover=0
