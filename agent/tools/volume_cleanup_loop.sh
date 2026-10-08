#!/usr/bin/env bash
# Reclaim leaked anonymous OSWorld VM volumes every 30 minutes (only delete dangling anonymous volumes; named/in-use volumes are left untouched).
PY=${SOFTWAREGYM_PYTHON:-/opt/miniconda3/envs/osworld/bin/python}
cd "$(dirname "$(readlink -f "$0")")/../.."
while true; do
  echo "===== $(date '+%F %T') prune ====="
  "$PY" -m agent.tools.cleanup --prune-volumes --json 2>&1 \
    | "$PY" -c "import sys,json;d=json.load(sys.stdin);v=d['anonymous_volumes'];print('reclaimable_before=%s removed=%d after=%s'%(v['reclaimable_before'],len(v['removed']),v['reclaimable_after']))" 2>/dev/null \
    || echo "(prune failed, skipping this round)"
  sleep 1800
done
