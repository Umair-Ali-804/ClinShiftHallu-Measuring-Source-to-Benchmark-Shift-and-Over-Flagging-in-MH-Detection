import json

import numpy as np
import pytest

from clinhallu.reporting.paired_bootstrap import (
    align_pair,
    compare_runs,
    load_predictions,
    paired_grouped_bootstrap,
)


def _write(path, ids, y, pred, score, groups=None):
    with path.open("w") as handle:
        for i, (idx, label, p, s) in enumerate(zip(ids, y, pred, score)):
            record = {"id": idx, "hallu_label": int(label), "prediction": int(p), "p_hallucinated_calibrated": float(s)}
            if groups is not None:
                record["group_key"] = groups[i]
            handle.write(json.dumps(record) + "\n")


def _systems(n=400, seed=0, a_noise=0.15, b_noise=0.35):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    def scores(noise):
        s = np.clip(y + rng.normal(0, noise * 3, size=n) * 0.5 + 0.0, -1, 2)
        return (s - s.min()) / (s.max() - s.min())
    sa, sb = scores(a_noise), scores(b_noise)
    return y, sa, sb


def test_identical_systems_have_zero_delta_and_ci_covering_zero(tmp_path):
    y, sa, _ = _systems()
    ids = [f"e{i}" for i in range(len(y))]
    _write(tmp_path / "a.jsonl", ids, y, sa >= 0.5, sa)
    _write(tmp_path / "b.jsonl", ids, y, sa >= 0.5, sa)
    result = compare_runs([tmp_path / "a.jsonl"], [tmp_path / "b.jsonl"], ["f1", "auroc"], n_bootstrap=200)
    for row in result:
        assert row["delta_a_minus_b"] == 0.0
        assert row["ci_low"] <= 0 <= row["ci_high"]
        assert not row["ci_excludes_zero"]


def test_clearly_better_system_has_positive_interval(tmp_path):
    y, sa, sb = _systems()
    ids = [f"e{i}" for i in range(len(y))]
    _write(tmp_path / "a.jsonl", ids, y, sa >= 0.5, sa)
    _write(tmp_path / "b.jsonl", ids, y, sb >= 0.5, sb)
    result = compare_runs([tmp_path / "a.jsonl"], [tmp_path / "b.jsonl"], ["f1", "auroc", "mcc"], n_bootstrap=300, seed=1)
    for row in result:
        assert row["delta_a_minus_b"] > 0 and row["ci_low"] > 0 and row["ci_excludes_zero"]
        assert row["p_two_sided"] < 0.05


def test_groups_are_resampled_jointly_and_seed_pairs_average(tmp_path):
    y, sa, sb = _systems(n=200)
    ids = [f"e{i}" for i in range(len(y))]
    groups = [f"g{i // 2}" for i in range(len(y))]
    pairs = []
    for seed in range(3):
        _write(tmp_path / f"a{seed}.jsonl", ids, y, sa >= 0.5, sa, groups)
        _write(tmp_path / f"b{seed}.jsonl", ids, y, sb >= 0.5, sb, groups)
        pairs.append(align_pair(load_predictions(tmp_path / f"a{seed}.jsonl"), load_predictions(tmp_path / f"b{seed}.jsonl")))
    one = paired_grouped_bootstrap(pairs[:1], "f1", n_bootstrap=50)
    three = paired_grouped_bootstrap(pairs, "f1", n_bootstrap=50)
    assert one["n_groups"] == 100 and three["n_seed_pairs"] == 3
    assert abs(one["delta_a_minus_b"] - three["delta_a_minus_b"]) < 1e-12


def test_misaligned_inputs_are_rejected(tmp_path):
    y, sa, _ = _systems(n=20)
    _write(tmp_path / "a.jsonl", [f"e{i}" for i in range(20)], y, sa >= 0.5, sa)
    _write(tmp_path / "b.jsonl", [f"x{i}" for i in range(20)], y, sa >= 0.5, sa)
    with pytest.raises(ValueError, match="different examples"):
        compare_runs([tmp_path / "a.jsonl"], [tmp_path / "b.jsonl"])
    with pytest.raises(ValueError, match="same number"):
        compare_runs([tmp_path / "a.jsonl"], [])
