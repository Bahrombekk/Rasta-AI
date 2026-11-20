#!/usr/bin/env bash
# Rasta AI — ma'lumotlarni zaxiralash (deployment hostida, Docker bilan).
#
# Zaxiralaydi:
#   - Docker volume'lari: rasta-catalog (Qdrant), rasta-images, rasta-accounts (SQLite)
#   - PostgreSQL (agar postgres konteyneri ishlayotgan bo'lsa) — pg_dump
#
# Foydalanish:
#   bash scripts/backup.sh                 # ./backups/<sana> ichiga
#   BACKUP_DIR=/mnt/disk bash scripts/backup.sh
#
# Tiklash (restore) bo'yicha ko'rsatmalar skript oxirida chiqadi.
set -euo pipefail

TS="$(date +%Y-%m-%d_%H%M%S)"
OUT="${BACKUP_DIR:-./backups}/${TS}"
mkdir -p "$OUT"
echo "[+] Zaxira papkasi: $OUT"

# --- Docker volume'larini topish (compose loyiha nomini prefiks qiladi) ---
backup_volume() {
  local suffix="$1"
  local vol
  vol="$(docker volume ls --format '{{.Name}}' | grep -E "${suffix}\$" | head -1 || true)"
  if [ -z "$vol" ]; then
    echo "[!] Volume topilmadi (${suffix}) — o'tkazib yuborildi"
    return 0
  fi
  echo "[+] Volume zaxiralanmoqda: $vol"
  docker run --rm -v "${vol}:/data:ro" -v "$(cd "$OUT" && pwd):/backup" \
    alpine tar czf "/backup/${suffix}.tgz" -C /data .
}

backup_volume "rasta-catalog"
backup_volume "rasta-images"
backup_volume "rasta-accounts"

# --- PostgreSQL (ixtiyoriy — konteyner ishlayotgan bo'lsa) ---
PG="$(docker ps --format '{{.Names}}' | grep -iE 'postgres' | head -1 || true)"
if [ -n "$PG" ]; then
  echo "[+] PostgreSQL dump: $PG"
  docker exec -t "$PG" pg_dump -U "${POSTGRES_USER:-rasta}" "${POSTGRES_DB:-rasta}" \
    | gzip > "$OUT/postgres.sql.gz"
else
  echo "[i] PostgreSQL konteyneri topilmadi — SQLite rejimi (rasta-accounts volume'da)"
fi

echo "[✓] Zaxira tayyor: $OUT"
ls -lh "$OUT"

cat <<EOF

--- TIKLASH (restore) ---
Volume (masalan katalog):
  docker run --rm -v <loyiha>_rasta-catalog:/data -v "\$(pwd)/${OUT}":/backup \\
    alpine sh -c "rm -rf /data/* && tar xzf /backup/rasta-catalog.tgz -C /data"
PostgreSQL:
  gunzip -c ${OUT}/postgres.sql.gz | docker exec -i <postgres_konteyner> \\
    psql -U rasta -d rasta
Tiklashdan oldin tegishli servislarni to'xtating (docker compose stop).
EOF
