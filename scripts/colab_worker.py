"""Entry point used by the Colab notebook: python scripts/colab_worker.py --drive-root ... --account A1"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinhallu.colab.worker import main  # noqa: E402

if __name__ == "__main__":
    main()
