#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT_DIR/docker-compose.yml}"
SERVICE="${SERVICE:-qq-api}"
RUN_BUILD=1
PULL_FLAG=""

usage() {
  cat <<USAGE
Usage: ${0##*/} [options]

Redeploy a docker-compose service (defaults to qq-api) while reusing build cache.

Options:
  -f, --compose-file PATH  Compose file to use (default: $COMPOSE_FILE)
  -s, --service NAME       Service name to redeploy (default: $SERVICE)
      --no-build           Skip docker compose build step
      --pull               Pass --pull to docker compose build
  -h, --help               Show this help message

Environment variables:
  COMPOSE_FILE  Override default compose file
  SERVICE       Override default service
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--compose-file)
      COMPOSE_FILE="$2"
      shift 2
      ;;
    -s|--service)
      SERVICE="$2"
      shift 2
      ;;
    --no-build)
      RUN_BUILD=0
      shift
      ;;
    --pull)
      PULL_FLAG="--pull"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Compose file not found: $COMPOSE_FILE" >&2
  exit 1
fi

cd "$ROOT_DIR"

if [[ $RUN_BUILD -eq 1 ]]; then
  echo "Building $SERVICE (cache enabled)..."
  docker compose -f "$COMPOSE_FILE" build $PULL_FLAG "$SERVICE"
else
  echo "Skipping build for $SERVICE"
fi

echo "Restarting $SERVICE via docker compose..."
docker compose -f "$COMPOSE_FILE" up -d "$SERVICE"

echo "Latest container status:"
docker compose -f "$COMPOSE_FILE" ps "$SERVICE"
