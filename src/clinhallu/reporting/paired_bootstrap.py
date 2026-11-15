"""Paired grouped bootstrap between two systems evaluated on the same examples.

Both systems are scored on identical rows; each bootstrap replicate resamples
question/context *groups* once and applies the same resample to A and B (and to
every seed pair), so the interval for ``metric(A) - metric(B)`` reflects test-set
sampling uncertainty rather than treating the two systems as independent.

    clinhallu compare \
        --a artifacts/adaptation/swap/seed_13/predictions/evaluation_predictions.jsonl \
            artifacts/adaptation/swap/seed_42/predictions/evaluation_predictions.jsonl \
        --b artifacts/adaptation/c5/seed_13/predictions/evaluation_predictions.jsonl \
            artifacts/adaptation/c5/seed_42/predictions/evaluation_predictions.jsonl \
        --metric f1 mcc auroc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _nan_if_single_class(fn):
    def wrapped(y, pred, score):
        if len(set(y.tolist())) < 2:
            return float("nan")
        return fn(y, pred, score)

    return wrapped


METRICS: dict[str, Callable] = {
    "f1": lambda y, p, s: f1_score(y, p, zero_division=0),
    "precision": lambda y, p, s: precision_score(y, p, zero_division=0),
    "recall": lambda y, p, s: recall_score(y, p, zero_division=0),
    "accuracy": lambda y, p, s: float((y == p).mean()),
    "mcc": _nan_if_single_class(lambda y, p, s: matthews_corrcoef(y, p)),
    "auroc": _nan_if_single_class(lambda y, p, s: roc_auc_score(y, s)),
    "auprc": _nan_if_single_class(lambda y, p, s: average_precision_score(y, s)),
}


def load_predictions(path: str | Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                key = str(record["id"])
                if key in rows:
                    raise ValueError(f"{path}: duplicate id {key}")
                rows[key] = record
    return rows


def align_pair(a: Mapping[str, dict], b: Mapping[str, dict]) -> dict[str, np.ndarray]:
    if set(a) != set(b):
        raise ValueError(
            f"Prediction files cover different examples ({len(set(a) ^ set(b))} ids differ)"
        )
    ids = sorted(a)
    labels_a = np.array([int(a[i]["hallu_label"]) for i in ids])
    labels_b = np.array([int(b[i]["hallu_label"]) for i in ids])
    if not np.array_equal(labels_a, labels_b):
        raise ValueError("Labels disagree between the two prediction files")
    groups = np.array([str(a[i].get("group_key", i)) for i in ids])
    return {
        "ids": np.array(ids),
        "y": labels_a,
        "pred_a": np.array([int(a[i]["prediction"]) for i in ids]),
        "score_a": np.array([float(a[i]["p_hallucinated_calibrated"]) for i in ids]),
        "pred_b": np.array([int(b[i]["prediction"]) for i in ids]),
        "score_b": np.array([float(b[i]["p_hallucinated_calibrated"]) for i in ids]),
        "groups": groups,
    }


def paired_grouped_bootstrap(
    pairs: Sequence[Mapping[str, np.ndarray]],
    metric: str = "f1",
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap ``mean over seed pairs of metric(A) - metric(B)``."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(METRICS)}")
    if not pairs:
        raise ValueError("Need at least one aligned prediction pair")
    fn = METRICS[metric]
    first_groups = pairs[0]["groups"]
    for pair in pairs[1:]:
        if not np.array_equal(pair["groups"], first_groups):
            raise ValueError("All seed pairs must be aligned to the same examples")
    unique, inverse = np.unique(first_groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]

    def delta_on(index: np.ndarray) -> tuple[float, float, float]:
        a_vals, b_vals = [], []
        for pair in pairs:
            y = pair["y"][index]
            a_vals.append(fn(y, pair["pred_a"][index], pair["score_a"][index]))
            b_vals.append(fn(y, pair["pred_b"][index], pair["score_b"][index]))
        a_mean, b_mean = float(np.mean(a_vals)), float(np.mean(b_vals))
        return a_mean, b_mean, a_mean - b_mean

    everything = np.arange(len(first_groups))
    point_a, point_b, point_delta = delta_on(everything)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap)
    for replicate in range(n_bootstrap):
        sampled = rng.integers(0, len(members), size=len(members))
        index = np.concatenate([members[g] for g in sampled])
        deltas[replicate] = delta_on(index)[2]
    deltas = deltas[~np.isnan(deltas)]
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(deltas, [alpha, 1.0 - alpha])
    p_value = float(min(1.0, 2.0 * min((deltas >= 0).mean(), (deltas <= 0).mean())))
    return {
        "metric": metric,
        "n_seed_pairs": len(pairs),
        "n_examples": int(len(first_groups)),
        "n_groups": int(len(members)),
        "n_bootstrap_used": int(len(deltas)),
        "a": point_a,
        "b": point_b,
        "delta_a_minus_b": point_delta,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_two_sided": p_value,
        "ci_excludes_zero": bool(low > 0 or high < 0),
    }


def compare_runs(
    a_paths: Sequence[str | Path],
    b_paths: Sequence[str | Path],
    metrics: Sequence[str] = ("f1", "mcc", "auroc"),
    **kwargs,
) -> list[dict]:
    if len(a_paths) != len(b_paths):
        raise ValueError("--a and --b need the same number of files (one per seed, same order)")
    pairs = [align_pair(load_predictions(a), load_predictions(b)) for a, b in zip(a_paths, b_paths)]
    return [paired_grouped_bootstrap(pairs, metric, **kwargs) for metric in metrics]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--a", nargs="+", required=True, help="Prediction files for system A (per seed)")
    parser.add_argument("--b", nargs="+", required=True, help="Prediction files for system B (same seed order)")
    parser.add_argument("--metric", nargs="+", default=["f1", "mcc", "auroc"], choices=sorted(METRICS))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="Optional JSON output path")
    args = parser.parse_args(argv)
    results = compare_runs(
        args.a, args.b, args.metric, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    text = json.dumps(results, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
