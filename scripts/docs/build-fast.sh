#!/usr/bin/env bash
# Local static-export iteration only — skips types:check and catalog drift checks.
# CI/production deploy continues to use scripts/docs/build.sh (see deploy-docs.yml).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCS_ROOT="${REPO_ROOT}/docs/fumadocs"

cd "${DOCS_ROOT}"
npm run build:fast
