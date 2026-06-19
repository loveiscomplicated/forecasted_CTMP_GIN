#!/usr/bin/env bash
# Vast.ai auto-termination helper.
#
# Usage:
#   bash scripts/vast_terminate.sh
#
# Expected environment:
#   CONTAINER_API_KEY   Vast.ai instance-scoped API key (injected via --env at create).
#   VAST_CONTAINERLABEL Vast-provided container label, typically "C.<instance_id>".
#   VAST_INSTANCE_ID    Fallback: explicit instance id if the label is unavailable.
#   DRY_RUN             If "1", echo the vastai command instead of executing it.
#   SEND_MESSAGE_PY     Optional path to send_message.py for Discord notify.
#   BOT_NAME            Optional bot name for notify.
#
# Exit behavior:
#   - On success: vastai stop instance <id>, then exit 0 (container will be torn down).
#   - On any failure/missing info: notify and hold forever (no destroy fallback by design).

set -uo pipefail

ts() { date '+%Y-%m-%d %H:%M:%S'; }

DRY_RUN="${DRY_RUN:-0}"
SEND_MESSAGE_PY="${SEND_MESSAGE_PY:-}"
BOT_NAME="${BOT_NAME:-vast_terminate}"

notify() {
  local msg="$1"
  echo "[$(ts)] $msg"
  if [[ -n "$SEND_MESSAGE_PY" && -f "$SEND_MESSAGE_PY" ]]; then
    python "$SEND_MESSAGE_PY" "$msg" "$BOT_NAME" || true
  fi
}

hold_forever() {
  echo "[$(ts)] holding forever..."
  while true; do sleep 3600; done
}

resolve_instance_id() {
  if [[ -n "${VAST_CONTAINERLABEL:-}" && "$VAST_CONTAINERLABEL" =~ ^C\.([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi
  if [[ -n "${VAST_INSTANCE_ID:-}" ]]; then
    echo "$VAST_INSTANCE_ID"
    return 0
  fi
  return 1
}

ensure_vastai_cli() {
  if command -v vastai >/dev/null 2>&1; then
    return 0
  fi
  echo "[$(ts)] vastai CLI not found, installing via pip..."
  if ! python -m pip install --quiet vastai; then
    return 1
  fi
  command -v vastai >/dev/null 2>&1
}

main() {
  echo "[$(ts)] ===== vast_terminate start ====="
  echo "[$(ts)] DRY_RUN=$DRY_RUN"
  echo "[$(ts)] VAST_CONTAINERLABEL='${VAST_CONTAINERLABEL:-}'"
  echo "[$(ts)] VAST_INSTANCE_ID='${VAST_INSTANCE_ID:-}'"

  if ! ensure_vastai_cli; then
    notify "[TERMINATE_SKIP] vastai CLI install failed. Holding without shutdown."
    hold_forever
  fi
  echo "[$(ts)] vastai: $(command -v vastai)"
  vastai --version 2>/dev/null || true

  if [[ -z "${CONTAINER_API_KEY:-}" ]]; then
    notify "[TERMINATE_SKIP] CONTAINER_API_KEY not set. Holding without shutdown."
    echo "[$(ts)] hint: pass via 'vastai create instance ... --env \"-e CONTAINER_API_KEY=<key>\"'"
    hold_forever
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    vastai set api-key "$CONTAINER_API_KEY" >/dev/null 2>&1 || {
      notify "[TERMINATE_SKIP] vastai set api-key failed. Holding without shutdown."
      hold_forever
    }
  fi

  local instance_id
  if ! instance_id="$(resolve_instance_id)"; then
    notify "[TERMINATE_SKIP] Could not resolve instance id. Holding without shutdown."
    echo "[$(ts)] env dump (vast|container):"
    env | grep -iE 'vast|container' || true
    hold_forever
  fi
  echo "[$(ts)] resolved instance_id=$instance_id"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$(ts)] [DRY_RUN] would run: vastai stop instance $instance_id"
    exit 0
  fi

  echo "[$(ts)] stopping instance $instance_id ..."
  if vastai stop instance "$instance_id"; then
    notify "[TERMINATE_OK] vastai stop instance $instance_id issued."
    exit 0
  fi

  notify "[TERMINATE_FAIL] vastai stop instance $instance_id failed. Holding without shutdown."
  hold_forever
}

main "$@"
