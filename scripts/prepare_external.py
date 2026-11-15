"""Convert a HaluBench-style file into the canonical external evaluation set.

    python scripts/prepare_external.py --source halubench.parquet \
        --out data/external/halubench_pubmedqa.jsonl

Defaults match the HaluBench PubMedQA slice (columns passage/question/answer/label,
label PASS/FAIL, source_ds == pubmedQA).  Check the column names against the file
you downloaded and the license on its dataset page; override with the flags below.
Rows whose question/context also occur in MedHallu train/val/test are dropped.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinhallu.data.external import convert_external  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default=str(ROOT / "data/external/halubench_pubmedqa.jsonl"))
    parser.add_argument("--question-col", default="question")
    parser.add_argument("--context-col", default="passage")
    parser.add_argument("--answer-col", default="answer")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--filter-col", default="source_ds")
    parser.add_argument("--filter-value", default="pubmedQA")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raw = ROOT / "data" / "raw"
    card = convert_external(
        args.source,
        args.out,
        question_col=args.question_col,
        context_col=args.context_col,
        answer_col=args.answer_col,
        label_col=args.label_col,
        filter_col=None if args.no_filter else args.filter_col,
        filter_value=None if args.no_filter else args.filter_value,
        exclude_files=[raw / "train.jsonl", raw / "val.jsonl", raw / "eval_data.jsonl"],
        seed=args.seed,
    )
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
