"""Build the unseen-generator evaluation set with an LLM that did not write MedHallu.

    export OPENROUTER_API_KEY=...        # Colab: put it in Secrets, never in the notebook
    python scripts/build_unseen_generator_set.py \
        --abstracts data/external/heldout_abstracts.jsonl \
        --model <vendor/model-id> --n 500

``--abstracts`` is JSONL with ``question`` and ``context`` (a PubMed abstract) that you
collected yourself; pairs overlapping MedHallu train/val/test are skipped.  Choose a
model different from the one that generated MedHallu.  Labels are by construction;
hand-check the audit CSV (default 50 rows) and report the error rate.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinhallu.api.openrouter import OpenRouterClient, ResponseCache  # noqa: E402
from clinhallu.data.external import build_unseen_set  # noqa: E402
from clinhallu.data.io import read_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--abstracts", required=True)
    parser.add_argument("--model", required=True, help="Explicit OpenRouter id, e.g. vendor/model")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out", default=str(ROOT / "data/external/unseen_generator.jsonl"))
    parser.add_argument("--cache-dir", default=str(ROOT / "data/external/.llm_cache"))
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    raw = ROOT / "data" / "raw"
    card = build_unseen_set(
        read_jsonl(args.abstracts),
        client=OpenRouterClient(rate_limit_retries=3),
        cache=ResponseCache(args.cache_dir),
        model=args.model,
        out_path=args.out,
        exclude_files=[raw / "train.jsonl", raw / "val.jsonl", raw / "eval_data.jsonl"],
        limit=args.n,
        settings={"temperature": args.temperature},
    )
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
