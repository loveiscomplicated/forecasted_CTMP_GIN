#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Postgres-on-container bootstrap for Optuna (single pod)
# - Installs PostgreSQL (Ubuntu/Debian)
# - Uses DEFAULT cluster location (/var/lib/postgresql/<ver>/main)
# - Starts cluster (even in containers where services don't auto-start)
# - Creates optuna user + database (idempotent)
# =========================================================

# -----------------------
# Config (override via env)
# -----------------------
PG_VER="${PG_VER:-14}"
CLUSTER_NAME="${CLUSTER_NAME:-main}"
PG_HOST="${PG_HOST:-127.0.0.1}"
PG_PORT="${PG_PORT:-5432}"

OPTUNA_USER="${OPTUNA_USER:-optuna}"
OPTUNA_PASS="${OPTUNA_PASS:-optuna_pw}"
OPTUNA_DB="${OPTUNA_DB:-optuna_db}"

# If you want to FORCE re-create the cluster (DANGEROUS: wipes DB), set:
#   RESET_CLUSTER=1
RESET_CLUSTER="${RESET_CLUSTER:-0}"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

need_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    die "must run as root (current uid=$(id -u))"
  fi
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

wait_ready() {
  for _ in {1..60}; do
    if su - postgres -c "pg_isready -h ${PG_HOST} -p ${PG_PORT}" 2>/dev/null | grep -q "accepting connections"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

# -----------------------
# Main
# -----------------------
need_root

export DEBIAN_FRONTEND=noninteractive

log "Adding PostgreSQL Official Repository for Ubuntu $(lsb_release -cs)..."
apt-get update -y
apt-get install -y gnupg2 wget lsb-release --no-install-recommends

# PostgreSQL 공식 GPG 키 추가 및 저장소 등록
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list

log "Installing PostgreSQL ${PG_VER} from PGDG..."
apt-get update -y
apt-get install -y --no-install-recommends \
  "postgresql-${PG_VER}" "postgresql-contrib-${PG_VER}" \
  postgresql-common postgresql-client-common \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Ensure cluster tools exist
have_cmd pg_lsclusters || die "pg_lsclusters not found (postgresql-common install failed?)"
have_cmd pg_ctlcluster || die "pg_ctlcluster not found"

# Optionally reset cluster (wipes data!)
if [[ "${RESET_CLUSTER}" == "1" ]]; then
  if pg_lsclusters | awk '{print $1" "$2}' | grep -q "^${PG_VER} ${CLUSTER_NAME}$"; then
    log "RESET_CLUSTER=1 -> dropping existing cluster ${PG_VER}/${CLUSTER_NAME} (THIS DELETES DATA)"
    pg_ctlcluster "${PG_VER}" "${CLUSTER_NAME}" stop || true
    pg_dropcluster --stop "${PG_VER}" "${CLUSTER_NAME}"
  fi
fi

# Create cluster if missing
if ! pg_lsclusters | awk '{print $1" "$2}' | grep -q "^${PG_VER} ${CLUSTER_NAME}$"; then
  log "Cluster ${PG_VER}/${CLUSTER_NAME} not found -> creating default cluster"
  pg_createcluster --port "${PG_PORT}" "${PG_VER}" "${CLUSTER_NAME}"
fi

# Start cluster
log "Starting cluster ${PG_VER}/${CLUSTER_NAME}..."
pg_ctlcluster "${PG_VER}" "${CLUSTER_NAME}" start || true

log "Waiting for Postgres to accept connections..."
if ! wait_ready; then
  log "pg_isready did not become ready. Showing diagnostics:"
  pg_lsclusters || true
  ps aux | grep -E "postgres|postmaster" || true
  ss -ltnp | grep ":${PG_PORT}" || true
  die "Postgres not accepting connections on ${PG_HOST}:${PG_PORT}"
fi
log "Postgres is accepting connections."

# Create role if not exists
log "Ensuring role '${OPTUNA_USER}' exists..."
su - postgres -c "psql -v ON_ERROR_STOP=1 -tAc \"SELECT 1 FROM pg_roles WHERE rolname='${OPTUNA_USER}'\" | grep -q 1 \
  || psql -v ON_ERROR_STOP=1 -c \"CREATE USER ${OPTUNA_USER} WITH PASSWORD '${OPTUNA_PASS}';\""

# Create database if not exists
log "Ensuring database '${OPTUNA_DB}' exists..."
su - postgres -c "psql -v ON_ERROR_STOP=1 -tAc \"SELECT 1 FROM pg_database WHERE datname='${OPTUNA_DB}'\" | grep -q 1 \
  || psql -v ON_ERROR_STOP=1 -c \"CREATE DATABASE ${OPTUNA_DB} OWNER ${OPTUNA_USER};\""

# Quick connection test as optuna user
log "Testing connection as '${OPTUNA_USER}'..."
PGPASSWORD="${OPTUNA_PASS}" psql -h "${PG_HOST}" -p "${PG_PORT}" -U "${OPTUNA_USER}" -d "${OPTUNA_DB}" -c "SELECT 1;" >/dev/null

log "Done."
log "Optuna storage URL:"
echo "postgresql+psycopg2://${OPTUNA_USER}:${OPTUNA_PASS}@${PG_HOST}:${PG_PORT}/${OPTUNA_DB}"