# run_batch.py
import subprocess
import shutil
import sys
from pathlib import Path
import json
import random
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# ---- paths -----------------------------------------------------------------
sim_file = Path("SimulatorDiscrete.py")
cfg_file = Path("profilingConfig.json")
logs_dir = Path("logs") 
logs_dir.mkdir(exist_ok=True)

# Write/overwrite a minimal config that disables interactive prompts
cfg_file.write_text(json.dumps({"non_interactive": True}), encoding="utf-8")

# Every file the simulator might create (add/remove as needed)
LOG_FILES = [
    "beta_gamma_log.csv",
    "alpha_log.csv",
    "beta_gamma_emp.csv",
    "beta_gamma_summary.txt",
    "log_mitigation.txt",
]

# ---- 10 independent runs ---------------------------------------------------
for run in range(1, 11):
    print(f"=== Run {run} ===")

    # Clean up leftovers from previous run
    for name in LOG_FILES:
        path = Path(name)
        if path.exists():
            path.unlink()

    # Launch the simulator (raise if it exits with error)
    subprocess.run([sys.executable, str(sim_file), str(cfg_file)], check=True)

    # Move any log that appeared into the logs/ directory with suffix _run
    for name in LOG_FILES:
        src = Path(name)
        if src.exists():
            dst = logs_dir / f"{src.stem}_{run}{src.suffix}"
            shutil.move(src, dst)

print("\nAll 10 runs finished. Check the 'logs/' directory for outputs.")
