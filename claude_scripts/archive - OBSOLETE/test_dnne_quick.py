#!/usr/bin/env python3
"""Quick test to show DNNE timeout"""

import subprocess
import time

cmd = """
source /home/asantanna/miniconda/bin/activate DNNE_PY38 && \
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO && \
python runner.py --headless
"""

print("Running DNNE Cartpole...")
start = time.time()

try:
    result = subprocess.run(
        ["bash", "-c", cmd],
        timeout=10,
        capture_output=True,
        text=True
    )
    print(f"Completed in {time.time() - start:.1f}s")
    print(f"Return code: {result.returncode}")
except subprocess.TimeoutExpired as e:
    print(f"TIMEOUT after {time.time() - start:.1f}s")
    print("Partial stdout:", e.stdout[:500] if e.stdout else "None")
    print("Partial stderr:", e.stderr[:500] if e.stderr else "None")