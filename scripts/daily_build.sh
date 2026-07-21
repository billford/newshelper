#!/usr/bin/env bash
# Wrapper invoked by launchd (com.billford.newshelper.daily.plist) twice a
# day on wanderlust. Runs the full build, then publishes on success. Never
# uses `set -e` -- failures are handled explicitly so a failed stage logs
# and notifies instead of leaving a half-finished state.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/daily_build.log"

mkdir -p "${LOG_DIR}"

log() {
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $1" >> "${LOG_FILE}"
}

notify_failure() {
  local stage="$1"
  log "FAILED at ${stage}"
  osascript -e "display notification \"NewsHelper ${stage} failed -- see ${LOG_FILE}\" with title \"NewsHelper build failed\"" >/dev/null 2>&1 || true
}

cd "${REPO_ROOT}"
log "=== build starting ==="

export PYTHONPATH="${REPO_ROOT}/src"
export NEWSHELPER_OLLAMA_MODEL="${NEWSHELPER_OLLAMA_MODEL:-qwen2.5:32b}"

if ! "${VENV_PYTHON}" -m newshelper.build >> "${LOG_FILE}" 2>&1; then
  notify_failure "build"
  exit 1
fi
log "build succeeded"

if ! bash "${REPO_ROOT}/scripts/publish.sh" >> "${LOG_FILE}" 2>&1; then
  notify_failure "publish"
  exit 1
fi
log "publish succeeded"

log "=== build complete ==="
