"""Build the revision training sets (S2K, S5K, SWAP, TGT4K, STRICT) from C1/C5.

Run once from the project root, before any new training run:

    python scripts/make_manifests.py

Writes data/raw/<cond>/train.jsonl (+ data/raw/strict/val.jsonl) and ID manifests
under data/splits/.  Re-running with the same seed reproduces identical files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinhallu.data.revision_splits import DEFAULT_SEED, build_revision_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    raw = args.root / "data" / "raw"
    summary = build_revision_splits(
        c1_path=raw / "c1" / "train.jsonl",
        c5_path=raw / "train.jsonl",
        val_path=raw / "val.jsonl",
        test_path=raw / "eval_data.jsonl",
        raw_dir=raw,
        splits_dir=args.root / "data" / "splits",
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
