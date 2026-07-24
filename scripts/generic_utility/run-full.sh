#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
PYTHON=${SG_PYTHON:-python}
OUT=${1:-$ROOT/log/generic_utility/full}
shift 2>/dev/null || true
export PYTHONPATH="$ROOT"
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON" "$ROOT/VERIFY_GENERIC_UTILITY_CAMPAIGN.py" --out "$OUT" "$@"
