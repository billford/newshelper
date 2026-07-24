#!/usr/bin/env bash
# Squashes gh-pages down to a single commit, dropping history (and every old
# video binary it carries) while keeping the branch's current content
# untouched. Run periodically (e.g. weekly, via a second launchd job) --
# publish.sh itself intentionally builds on top of gh-pages history each
# run, so without this the branch grows a few MB/day forever.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/squash_gh_pages.log"
BRANCH="gh-pages"

mkdir -p "${LOG_DIR}"

log() {
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $1" >> "${LOG_FILE}"
}

cd "${REPO_ROOT}"
log "=== squash starting ==="

if ! git show-ref --verify --quiet "refs/remotes/origin/${BRANCH}"; then
  log "no ${BRANCH} branch on origin yet; nothing to squash"
  exit 0
fi

WORKTREE_DIR="$(mktemp -d)"
trap 'rm -rf "${WORKTREE_DIR}"' EXIT

git worktree add "${WORKTREE_DIR}" "${BRANCH}" >> "${LOG_FILE}" 2>&1
cd "${WORKTREE_DIR}"

CURRENT_SHA="$(git rev-parse HEAD)"
git checkout --orphan "${BRANCH}-squash-tmp" >> "${LOG_FILE}" 2>&1
git add -A
git commit -m "Squash ${BRANCH} history (was ${CURRENT_SHA})" >> "${LOG_FILE}" 2>&1
git branch -M "${BRANCH}-squash-tmp" "${BRANCH}"
git push origin "${BRANCH}" --force >> "${LOG_FILE}" 2>&1

cd "${REPO_ROOT}"
git worktree remove "${WORKTREE_DIR}" --force
log "squash complete"
log "=== squash complete ==="
