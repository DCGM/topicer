#. .deploy/.env
#export BACKEND_URL
export BRANCH_NAME=main
export NOW=$(date --iso-8601=seconds)

docker compose -f compose.yaml $1 $2 $3
