#!/usr/bin/env bash
# Copies the RAG index to lampoon, where rag_serve (newshelper-rag-serve.service)
# answers retrieval for the chatbot proxy. Run after every build by
# daily_build.sh.
#
#   scripts/sync_rag_to_lampoon.sh          # index only
#   scripts/sync_rag_to_lampoon.sh --code   # also the retrieval modules + config
#
# Code/config changes only take effect after
# `sudo systemctl restart newshelper-rag-serve` on lampoon; index changes are
# picked up on the next query (VectorStore opens the table per query).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="lampoon"
REMOTE_DIR="newshelper-retrieval"
# BatchMode: under launchd there is nobody to answer a prompt, so fail fast.
SSH_CMD="ssh -o BatchMode=yes -o ConnectTimeout=10"

cd "${REPO_ROOT}"

if [ "${1:-}" = "--code" ]; then
  ${SSH_CMD} "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}/src/newshelper ${REMOTE_DIR}/config ${REMOTE_DIR}/data" || exit 1
  rsync -a -e "${SSH_CMD}" \
    src/newshelper/__init__.py \
    src/newshelper/rag_config.py \
    src/newshelper/rag_embed.py \
    src/newshelper/rag_retrieve.py \
    src/newshelper/rag_serve.py \
    src/newshelper/rag_store.py \
    "${REMOTE_HOST}:${REMOTE_DIR}/src/newshelper/" || exit 1
  rsync -a -e "${SSH_CMD}" scripts/lampoon-newshelper-rag.yaml \
    "${REMOTE_HOST}:${REMOTE_DIR}/config/rag.yaml" || exit 1
fi

# The index is live while this runs. LanceDB's _versions manifests sort
# before data/, so a plain rsync can land a manifest before the files it
# names; --delay-updates renames everything into place at the end instead,
# and --delete-after keeps old files until the new ones are there.
rsync -a --delete-after --delay-updates -e "${SSH_CMD}" \
  data/rag_store/ "${REMOTE_HOST}:${REMOTE_DIR}/data/rag_store/"
