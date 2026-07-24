#!/usr/bin/env bash
# Offsite backup of the buckets whose loss would kill thesis reproducibility:
# bronze snapshots and the DVC remote. Delta/Lance are rebuildable and are
# deliberately NOT backed up. Dry-run by default; APPLY=1 to sync for real.
set -euo pipefail

: "${BACKUP_REMOTE:?set BACKUP_REMOTE (an rclone remote:bucket) in .env}"
: "${S3_ENDPOINT_URL:?set S3_ENDPOINT_URL in .env}"

FLAGS=(--s3-endpoint "$S3_ENDPOINT_URL" --checksum)
if [ "${APPLY:-0}" != "1" ]; then
  echo "dry-run (set APPLY=1 to sync for real)"
  FLAGS+=(--dry-run)
fi

for bucket in sda-bronze sda-dvc; do
  echo "== $bucket -> $BACKUP_REMOTE/$bucket"
  rclone sync ":s3:$bucket" "$BACKUP_REMOTE/$bucket" "${FLAGS[@]}"
done
