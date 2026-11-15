"""Write jobs/jobs.json and per-account preferred lists (jobs/jobs_A1.txt ...)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinhallu.colab.jobs import write_jobs  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(write_jobs(ROOT / "jobs"), indent=2))
