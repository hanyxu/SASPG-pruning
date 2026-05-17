#!/usr/bin/env bash
set -euo pipefail

echo "ERROR: SUPERB downstream (s3prl) was removed from this release. Use upstream only:" >&2
echo "  python3 -m core experiments run --exp-id …" >&2
echo "  python3 -m core pipeline --run-upstream-first --exp-id …" >&2
exit 2
