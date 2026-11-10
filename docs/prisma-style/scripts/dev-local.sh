#!/usr/bin/env bash
# Run the Prisma-style docs site from a local-SSD mirror.
# The mirror holds docs/fumadocs (shared content and widgets, without its own
# node_modules) next to docs/prisma-style, so ../fumadocs resolves the same way
# it does in the checkout and every shared module uses this site's packages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_SRC="$(cd "${SCRIPT_DIR}/.." && pwd)"
FUMADOCS_SRC="$(cd "${SITE_SRC}/../fumadocs" && pwd)"
MIRROR_ID="$(node -e 'process.stdout.write(require("node:crypto").createHash("sha256").update(require("node:fs").realpathSync(process.argv[1])).digest("hex").slice(0,16))' "${SITE_SRC}")"
LOCAL_ROOT="${WF_PRISMA_LOCAL_ROOT:-/tmp/wf-prisma-style/${MIRROR_ID}}"
LOCAL_DOCS="${LOCAL_ROOT}/docs"
LOCAL_SITE="${LOCAL_DOCS}/prisma-style"
PORT="${PORT:-3002}"

RSYNC_EXCLUDES=(
  --exclude 'node_modules'
  --exclude '.next'
  --exclude '.next.*'
  --exclude 'out'
  --exclude 'out.*'
  --exclude '/tmp/'
  --exclude '.source/'
)

ensure_node() {
  if [[ -n "${WF_DOCS_NODE_BIN:-}" ]]; then
    export PATH="${WF_DOCS_NODE_BIN}:${PATH}"
  fi
  command -v npm >/dev/null 2>&1 || {
    echo "npm is required; install Node.js or set WF_DOCS_NODE_BIN" >&2
    exit 1
  }
}

sync_sources() {
  mkdir -p "${LOCAL_DOCS}/fumadocs" "${LOCAL_SITE}"
  rsync -a --delete "${RSYNC_EXCLUDES[@]}" "${FUMADOCS_SRC}/" "${LOCAL_DOCS}/fumadocs/"
  rsync -a --delete "${RSYNC_EXCLUDES[@]}" --exclude '/app/shared/' "${SITE_SRC}/" "${LOCAL_SITE}/"
  # Shared files resolve packages from this site's install, for webpack and tsc alike.
  ln -sfn ../prisma-style/node_modules "${LOCAL_DOCS}/fumadocs/node_modules"
}

install_deps() {
  (
    cd "${LOCAL_SITE}"
    if [[ ! -d node_modules/next ]]; then
      echo "Installing prisma-style dependencies on local SSD"
      if [[ -f package-lock.json ]]; then
        npm ci --prefer-offline --no-audit --no-fund
      else
        npm install --prefer-offline --no-audit --no-fund
        cp package-lock.json "${SITE_SRC}/package-lock.json"
      fi
    fi
  )
}

run_dev() {
  ensure_node
  sync_sources
  install_deps
  cd "${LOCAL_SITE}"
  node scripts/predev.mjs
  local host_ip
  host_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  export WF_DOCS_DEV_ORIGINS="${WF_DOCS_DEV_ORIGINS:-${host_ip}}"
  echo "Starting prisma-style dev server from ${LOCAL_SITE} on port ${PORT}"
  exec npx --no-install next dev --hostname 0.0.0.0 --webpack --port "${PORT}" "$@"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  dev    Sync both sites to local SSD, install dependencies, and start next dev (default)
  sync   Re-sync sources into the running mirror (hot reload picks the changes up)
  path   Print the mirror directory

Environment:
  WF_PRISMA_LOCAL_ROOT  Mirror root (default: /tmp/wf-prisma-style/<checkout-id>)
  WF_DOCS_NODE_BIN      Directory containing node/npm executables
  WF_DOCS_DEV_ORIGINS   Extra hostnames allowed to reach dev resources (default: this host's IP)
  PORT                  Dev server port (default: 3002)
EOF
}

case "${1:-dev}" in
  dev) shift || true; run_dev "$@" ;;
  sync) ensure_node; sync_sources; (cd "${LOCAL_SITE}" && node scripts/sync-shared-styles.mjs) ;;
  path) echo "${LOCAL_SITE}" ;;
  -h|--help|help) usage ;;
  *)
    echo "Unknown command: $1" >&2
    usage >&2
    exit 1
    ;;
esac
