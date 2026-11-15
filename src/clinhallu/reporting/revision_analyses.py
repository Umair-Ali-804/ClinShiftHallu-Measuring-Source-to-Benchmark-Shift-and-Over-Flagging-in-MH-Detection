"""CPU-only analyses computed from saved prediction files.

    clinhallu analyze claim-count   --pred <predictions.jsonl> [--out file.json]
    clinhallu analyze low-prevalence --pred <predictions.jsonl> --counts 26 56 125
    clinhallu analyze recalibrate   --val <validation_or_other_preds.jsonl> --test <preds.jsonl>

Every function takes plain records so it works on any run (C1..C5, STRICT, SWAP,
judge baselines converted to the same schema).  Nothing here retrains a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score


def read_records(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------- #
# 1. Claim-count vs false-positive rate
# --------------------------------------------------------------------------- #
def claim_count_false_positive_rate(
    records: Sequence[dict], bins: Sequence[int] = (1, 2, 3, 4)
) -> dict:
    """FP rate on SUPPORTED answers (label 0) as a function of claim count.

    ``bins=(1,2,3,4)`` yields buckets 1, 2, 3 and 4+.  Also reports a logistic
    trend (log-odds of a false positive per extra claim) with a grouped-bootstrap
    free analytic slope, and the answer-length correlation, because claim count and
    length are confounded.
    """
    supported = [r for r in records if int(r["hallu_label"]) == 0]
    if not supported or "claim_count" not in supported[0]:
        raise ValueError("Predictions lack claim_count; re-run evaluation with this version")
    claims = np.array([int(r["claim_count"]) for r in supported])
    fp = np.array([int(r["prediction"]) == 1 for r in supported], dtype=float)
    words = np.array([int(r["answer_words"]) for r in supported], dtype=float)
    rows = []
    for index, low in enumerate(bins):
        high = bins[index + 1] if index + 1 < len(bins) else None
        mask = claims >= low if high is None else (claims >= low) & (claims < high)
        label = f"{low}+" if high is None else (str(low) if high == low + 1 else f"{low}-{high - 1}")
        n = int(mask.sum())
        rows.append(
            {
                "claims": label,
                "n_supported": n,
                "false_positive_rate": float(fp[mask].mean()) if n else None,
                "mean_answer_words": float(words[mask].mean()) if n else None,
            }
        )
    slope = None
    if len(set(claims.tolist())) > 1 and 0 < fp.sum() < len(fp):
        model = LogisticRegression(C=1e6, max_iter=1000).fit(claims.reshape(-1, 1), fp)
        slope = float(model.coef_[0][0])
    corr = float(np.corrcoef(claims, words)[0, 1]) if claims.std() > 0 and words.std() > 0 else None
    return {
        "n_supported": len(supported),
        "overall_false_positive_rate": float(fp.mean()),
        "buckets": rows,
        "logit_slope_per_claim": slope,
        "corr_claims_vs_answer_words": corr,
        "note": "claim_count is a sentence-level proxy; confirm on a hand-checked sample.",
    }


# --------------------------------------------------------------------------- #
# 2. Real low-prevalence subsampling
# --------------------------------------------------------------------------- #
def low_prevalence_resampling(
    records: Sequence[dict],
    hallucinated_counts: Sequence[int] = (26, 56, 125),
    repeats: int = 100,
    seed: int = 0,
) -> dict:
    """Keep every supported answer, sample k hallucinated ones, repeat.

    Compares the empirical precision/F1 with the Bayes projection computed from
    the full-set recall and false-positive rate at the same prevalence.
    """
    y = np.array([int(r["hallu_label"]) for r in records])
    pred = np.array([int(r["prediction"]) for r in records])
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    tpr = float(pred[positives].mean())
    fpr = float(pred[negatives].mean())
    rng = np.random.default_rng(seed)
    out = []
    for k in hallucinated_counts:
        if k > len(positives):
            raise ValueError(f"Cannot sample {k} hallucinated rows from {len(positives)}")
        precisions, f1s = [], []
        for _ in range(repeats):
            chosen = np.concatenate([negatives, rng.choice(positives, size=k, replace=False)])
            precisions.append(precision_score(y[chosen], pred[chosen], zero_division=0))
            f1s.append(f1_score(y[chosen], pred[chosen], zero_division=0))
        prevalence = k / (k + len(negatives))
        tp, fp = tpr * k, fpr * len(negatives)
        projected_precision = tp / (tp + fp) if tp + fp else 0.0
        projected_f1 = 2 * tp / (2 * tp + fp + (1 - tpr) * k) if tp else 0.0
        out.append(
            {
                "hallucinated_sampled": k,
                "supported_kept": int(len(negatives)),
                "prevalence": prevalence,
                "precision_mean": float(np.mean(precisions)),
                "precision_ci95": [float(np.quantile(precisions, 0.025)), float(np.quantile(precisions, 0.975))],
                "f1_mean": float(np.mean(f1s)),
                "f1_ci95": [float(np.quantile(f1s, 0.025)), float(np.quantile(f1s, 0.975))],
                "bayes_projected_precision": float(projected_precision),
                "bayes_projected_f1": float(projected_f1),
            }
        )
    return {"full_set_recall": tpr, "full_set_false_positive_rate": fpr, "repeats": repeats, "rows": out}


# --------------------------------------------------------------------------- #
# 3. Calibration + threshold refit (fair judge comparison / STRICT decomposition)
# --------------------------------------------------------------------------- #
def _scores(records: Sequence[dict], key: str) -> np.ndarray:
    if key == "logit" and "logit" in records[0]:
        return np.array([float(r["logit"]) for r in records])
    p = np.clip(np.array([float(r[key]) for r in records]), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def recalibrate(
    fit_records: Sequence[dict],
    eval_records: Sequence[dict],
    score_key: str = "logit",
) -> dict:
    """Fit Platt scaling and an F1-optimal threshold on ``fit_records`` only, then
    apply both, frozen, to ``eval_records``.

    ``fit_records`` must never be the evaluation split.  Use it to (a) give LLM
    judges calibrated, tuned thresholds before comparing over-flagging, and (b)
    refit a STRICT run on target validation to separate checkpoint-selection
    leakage from calibration leakage.
    """
    y_fit = np.array([int(r["hallu_label"]) for r in fit_records])
    x_fit = _scores(fit_records, score_key).reshape(-1, 1)
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(x_fit, y_fit)
    p_fit = platt.predict_proba(x_fit)[:, 1]
    grid = np.unique(np.concatenate([np.linspace(0.01, 0.99, 197), p_fit]))
    best = max(grid, key=lambda t: f1_score(y_fit, (p_fit >= t).astype(int), zero_division=0))
    y_eval = np.array([int(r["hallu_label"]) for r in eval_records])
    p_eval = platt.predict_proba(_scores(eval_records, score_key).reshape(-1, 1))[:, 1]
    pred_eval = (p_eval >= best).astype(int)
    original = np.array([int(r["prediction"]) for r in eval_records])
    return {
        "platt_coef": float(platt.coef_[0][0]),
        "platt_intercept": float(platt.intercept_[0]),
        "threshold": float(best),
        "fit_f1": float(f1_score(y_fit, (p_fit >= best).astype(int), zero_division=0)),
        "eval": {
            "f1": float(f1_score(y_eval, pred_eval, zero_division=0)),
            "precision": float(precision_score(y_eval, pred_eval, zero_division=0)),
            "recall": float(recall_score(y_eval, pred_eval, zero_division=0)),
            "flag_rate": float(pred_eval.mean()),
        },
        "eval_original_decisions": {
            "f1": float(f1_score(y_eval, original, zero_division=0)),
            "precision": float(precision_score(y_eval, original, zero_division=0)),
            "recall": float(recall_score(y_eval, original, zero_division=0)),
            "flag_rate": float(original.mean()),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    claim = sub.add_parser("claim-count")
    claim.add_argument("--pred", required=True)
    low = sub.add_parser("low-prevalence")
    low.add_argument("--pred", required=True)
    low.add_argument("--counts", nargs="+", type=int, default=[26, 56, 125])
    low.add_argument("--repeats", type=int, default=100)
    low.add_argument("--seed", type=int, default=0)
    recal = sub.add_parser("recalibrate")
    recal.add_argument("--val", required=True, help="Records to FIT on (never the test split)")
    recal.add_argument("--test", required=True, help="Records to evaluate on, frozen")
    recal.add_argument("--score-key", default="logit")
    for p in (claim, low, recal):
        p.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "claim-count":
        result = claim_count_false_positive_rate(read_records(args.pred))
    elif args.command == "low-prevalence":
        result = low_prevalence_resampling(
            read_records(args.pred), args.counts, args.repeats, args.seed
        )
    else:
        result = recalibrate(read_records(args.val), read_records(args.test), args.score_key)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
