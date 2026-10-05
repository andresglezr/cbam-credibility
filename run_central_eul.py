# -*- coding: utf-8 -*-
"""Solve the central family (recenter_compute.CENTRAL_ENV) under the suffix given as argv[1]."""
import subprocess, sys, time
from pathlib import Path
import recenter_compute as rc

SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "newblk_eul"
env = rc.clean_env(); env.update(rc.CENTRAL_ENV)
env["RAMSEY_OUT_SUFFIX"] = SUFFIX; env["RAMSEY_RUN_LABEL"] = f"central_{SUFFIX}"
rc.LOGS.mkdir(parents=True, exist_ok=True)
log = rc.LOGS / f"run_{SUFFIX}.log"
t0 = time.time()
with log.open("w", encoding="utf-8", errors="replace") as fh:
    rc_ = subprocess.run([str(rc.PYTHON), str(rc.MODEL)], cwd=str(rc.ROOT), env=env,
                         stdout=fh, stderr=subprocess.STDOUT, text=True).returncode
ok = all(p.exists() and p.stat().st_size > 0 for p in rc.expected_outputs(SUFFIX))
print(f"central {SUFFIX}: rc={rc_} outputs_complete={ok} elapsed={(time.time()-t0)/60:.1f} min")
sys.exit(0 if (rc_ == 0 and ok) else 1)
