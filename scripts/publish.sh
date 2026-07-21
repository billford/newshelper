#!/usr/bin/env bash
# Publishes dist/ to the gh-pages branch. Run on wanderlust after build.py,
# from cron. Requires a local git remote with push access already
# configured (deploy key or PAT) -- not run from GitHub Actions, per
# ADR-001, since the model only needs to be reachable locally.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"
BRANCH="gh-pages"

if [ ! -d "${DIST_DIR}" ]; then
  echo "error: ${DIST_DIR} does not exist -- run 'python -m newshelper.build' first" >&2
  exit 1
fi

cd "${REPO_ROOT}"

WORKTREE_DIR="$(mktemp -d)"
trap 'rm -rf "${WORKTREE_DIR}"' EXIT

if git show-ref --verify --quiet "refs/remotes/origin/${BRANCH}"; then
  git worktree add "${WORKTREE_DIR}" "${BRANCH}"
else
  git worktree add --orphan -b "${BRANCH}" "${WORKTREE_DIR}"
fi

rsync -a --delete --exclude='.git' "${DIST_DIR}/" "${WORKTREE_DIR}/"

cd "${WORKTREE_DIR}"
git add -A
if git diff --cached --quiet; then
  echo "no changes to publish"
else
  git commit -m "Daily digest: $(date -u +%Y-%m-%d)"
  git push origin "${BRANCH}"
fi

cd "${REPO_ROOT}"
git worktree remove "${WORKTREE_DIR}" --force
