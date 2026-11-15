import json

import numpy as np
import pytest

from clinhallu.data.validation import validate_fixed_splits
from clinhallu.engine.example_metadata import answer_claims, example_metadata, source_tag
from clinhallu.engine.prediction_writer import write_predictions
from clinhallu.reporting.revision_analyses import (
    claim_count_false_positive_rate,
    low_prevalence_resampling,
    recalibrate,
)


def test_claim_counting_and_tags():
    assert len(answer_claims("The drug works. It lowers risk. Ok.")) == 2  # 'Ok.' is < 3 words
    assert len(answer_claims("single claim here")) == 1
    assert source_tag("c1r12") == "source" and source_tag("c3r4") == "target" and source_tag("valr1") == "target"
    meta = example_metadata(
        {"sample_id": "c1r1", "question": "Q one", "context": "a b c", "answer": "One two three. Four five six."},
    )
    assert meta["source_tag"] == "source" and meta["claim_count"] == 2 and meta["answer_words"] == 6
    assert meta["context_words"] == 3 and len(meta["group_key"]) == 16


def test_prediction_writer_logits_and_metadata(tmp_path):
    samples = [{"sample_id": "a", "question": "q", "context": "c", "answer": "x y z"}, {"sample_id": "b", "question": "q2", "context": "c2", "answer": "x y z"}]
    meta = [example_metadata(s) for s in samples]
    path = tmp_path / "p.jsonl"
    write_predictions(samples, [0, 1], [0.1, 0.8], [0.2, 0.9], [0, 1], path, logits=[-2.0, 1.5], metadata=meta)
    rows = [json.loads(l) for l in path.open()]
    assert rows[1]["logit"] == 1.5 and rows[0]["claim_count"] == 1 and rows[0]["id"] == "a"
    with pytest.raises(ValueError):
        write_predictions(samples, [0, 1], [0.1, 0.8], [0.2, 0.9], [0, 1], path, logits=[1.0])
    # still works the legacy way
    write_predictions(samples, [0, 1], [0.1, 0.8], [0.2, 0.9], [0, 1], path)
    assert "logit" not in json.loads(path.read_text().splitlines()[0])


def _preds(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = i % 2
        claims = int(rng.integers(1, 6))
        logit = (2.0 if y else -2.0) * 0.6 + rng.normal(0, 1.2) + (0.35 * claims if y == 0 else 0)
        p = 1 / (1 + np.exp(-logit))
        rows.append({"id": f"e{i}", "hallu_label": y, "logit": float(logit), "p_hallucinated_calibrated": float(p),
                     "prediction": int(p >= 0.5), "claim_count": claims, "answer_words": 8 * claims + int(rng.integers(0, 5))})
    return rows


def test_claim_count_analysis_detects_the_built_in_trend():
    result = claim_count_false_positive_rate(_preds())
    rates = [b["false_positive_rate"] for b in result["buckets"] if b["n_supported"]]
    assert rates[-1] > rates[0]
    assert result["logit_slope_per_claim"] > 0
    with pytest.raises(ValueError):
        claim_count_false_positive_rate([{"hallu_label": 0, "prediction": 0}])


def test_low_prevalence_matches_bayes_projection():
    result = low_prevalence_resampling(_preds(2000), (26, 100), repeats=200, seed=1)
    for row in result["rows"]:
        assert abs(row["precision_mean"] - row["bayes_projected_precision"]) < 0.05
    assert result["rows"][0]["precision_mean"] < result["rows"][1]["precision_mean"]
    with pytest.raises(ValueError):
        low_prevalence_resampling(_preds(100), (500,))


def test_recalibration_uses_fit_split_only():
    fit, evaluate = _preds(1000, 1), _preds(1000, 2)
    for r in fit:  # make the fit split badly mis-thresholded
        r["prediction"] = 1
    out = recalibrate(fit, evaluate)
    assert 0.0 < out["threshold"] < 1.0
    assert out["eval"]["f1"] > 0.6
    # the evaluation labels never influence the fitted parameters
    shuffled = [dict(r, hallu_label=1 - r["hallu_label"]) for r in evaluate]
    assert recalibrate(fit, shuffled)["threshold"] == out["threshold"]


def _vrow(i, q, label=0):
    return {"id": i, "question": q, "context": f"ctx {q}", "answer": f"ans {i}", "hallu_label": label}


def _vwrite(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_overlap_policy_in_validation(tmp_path):
    train = [_vrow(f"t{i}", f"q{i}", i % 2) for i in range(10)]
    val = [_vrow(f"v{i}", f"v{i}", i % 2) for i in range(4)] + [_vrow("dup", "q3", 1)]
    test = [_vrow(f"e{i}", f"e{i}", i % 2) for i in range(4)]
    for name, rows in (("train", train), ("val", val), ("test", test)):
        _vwrite(tmp_path / f"{name}.jsonl", rows)
    args = (tmp_path / "train.jsonl", tmp_path / "val.jsonl", tmp_path / "test.jsonl")
    with pytest.raises(RuntimeError, match="leakage"):
        validate_fixed_splits(*args, overlap_policy="error")
    report = validate_fixed_splits(*args, overlap_policy="remove_lower_priority")
    assert report["overlap_group_counts"]["train__validation"] == 1
    # validation/test overlap is never tolerated
    test.append(_vrow("e9", "v1", 0))
    _vwrite(tmp_path / "test.jsonl", test)
    with pytest.raises(RuntimeError, match="validation and test"):
        validate_fixed_splits(*args, overlap_policy="remove_lower_priority")
