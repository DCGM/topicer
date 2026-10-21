#. .deploy/.env
#export BACKEND_URL
# A branch or a release tag, e.g. BRANCH_NAME=v0.1.0 ./update.sh build
export BRANCH_NAME=${BRANCH_NAME:-main}
export NOW=$(date --iso-8601=seconds)

docker compose -f compose.yaml $1 $2 $3
