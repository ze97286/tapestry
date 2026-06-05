#!/bin/bash
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
mbert_v06_init "$0"
mbert_v06_submit_dmr_prepare
