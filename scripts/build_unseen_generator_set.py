
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
