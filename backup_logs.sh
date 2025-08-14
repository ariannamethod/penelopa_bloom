#!/usr/bin/env bash
set -euo pipefail

# Usage: ./backup_logs.sh /secure/backup
# Copies lines.db and lines.txt from the logs directory to the specified backup
# location with a timestamp. Suitable for scheduling via cron.

LOG_DIR="$(dirname "$0")/logs"
DB_SRC="$LOG_DIR/lines.db"
TXT_SRC="$LOG_DIR/lines.txt"
DEST_DIR="${1:-/secure/backup}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$DEST_DIR"
[ -f "$DB_SRC" ] && cp "$DB_SRC" "$DEST_DIR/lines_${TIMESTAMP}.db"
[ -f "$TXT_SRC" ] && cp "$TXT_SRC" "$DEST_DIR/lines_${TIMESTAMP}.txt"
