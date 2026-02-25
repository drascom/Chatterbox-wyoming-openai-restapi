#!/usr/bin/env bash
set -euo pipefail

# Deep Docker cleanup utility.
# WARNING: this is destructive and can remove unrelated Docker data.
#
# Usage:
#   scripts/docker-clean.sh --yes
#   scripts/docker-clean.sh --yes --project chatterbox-wyoming-openai-restapi
#   scripts/docker-clean.sh --yes --all-volumes
#
# Options:
#   --yes            Required. Confirms destructive cleanup.
#   --project NAME   Optional. Also force-remove containers/images matching NAME.
#   --all-volumes    Optional. Removes ALL local Docker volumes.
#   --help           Show this help.

YES=0
PROJECT=""
ALL_VOLUMES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)
      YES=1
      shift
      ;;
    --project)
      PROJECT="${2:-}"
      shift 2
      ;;
    --all-volumes)
      ALL_VOLUMES=1
      shift
      ;;
    --help|-h)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ "$YES" -ne 1 ]]; then
  echo "Refusing to run without --yes"
  exit 1
fi

echo "==> Stopping/removing Compose stack in current directory (if any)"
docker compose down --remove-orphans --volumes || true

if [[ -n "$PROJECT" ]]; then
  echo "==> Removing project-matching containers/images for: $PROJECT"
  docker ps -a --format '{{.ID}} {{.Names}} {{.Image}}' | awk -v p="$PROJECT" '$0 ~ p {print $1}' | xargs -r docker rm -f
  docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | awk -v p="$PROJECT" '$0 ~ p {print $2}' | xargs -r docker rmi -f
fi

echo "==> Pruning stopped containers"
docker container prune -f

echo "==> Pruning dangling + unused images"
docker image prune -a -f

echo "==> Pruning unused networks"
docker network prune -f

if [[ "$ALL_VOLUMES" -eq 1 ]]; then
  echo "==> Pruning ALL unused volumes"
  docker volume prune -f
else
  echo "==> Keeping global volumes (pass --all-volumes to prune them)"
fi

echo "==> Pruning build cache"
docker builder prune -a -f

echo "==> Final system prune"
docker system prune -a -f

echo "Done."
