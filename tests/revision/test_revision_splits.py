import json
from pathlib import Path

import pytest

from clinhallu.data.io import read_jsonl
from clinhallu.data.revision_splits import build_revision_splits, stratified_nested_order

ROOT = Path(__file__).resolve().parents[2]


def _row(identifier, label, tag=None):
    return {
        "id": identifier,
        "question": f"question {tag or identifier}",
        "context": f"context {tag or identifier}",
        "answer": f"answer {identifier}",
        "hallu_label": label,
    }


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _toy(tmp_path, n_source=10000, n_target=40):
    source = [_row(f"c1r{i}", i % 2) for i in range(n_source)]
    target = [_row(f"c2r{i}", i % 2) for i in range(n_target)]
    val = [_row(f"valr{i}", i % 2) for i in range(20)]
    test = [_row(f"eval{i}", i % 2) for i in range(10)]
    raw = tmp_path / "raw"
    _write(raw / "c1" / "train.jsonl", source)
    # C5 order deliberately interleaves target rows into the source rows
    mixed = source[:3] + target + source[3:]
    _write(raw / "train.jsonl", mixed)
    _write(raw / "val.jsonl", val)
    _write(raw / "eval_data.jsonl", test)
    return raw


def _build(tmp_path, raw, **kw):
    return build_revision_splits(
        c1_path=raw / "c1" / "train.jsonl",
        c5_path=raw / "train.jsonl",
        val_path=raw / "val.jsonl",
        test_path=raw / "eval_data.jsonl",
        raw_dir=raw,
        splits_dir=tmp_path / "splits",
        **kw,
    )


def test_prefixes_are_balanced_and_nested():
    rows = [_row(f"c1r{i}", i % 2) for i in range(100)]
    order = stratified_nested_order(rows, seed=1)
    label = {r["id"]: r["hallu_label"] for r in rows}
    for size in (2, 10, 50, 100):
        assert sum(label[i] for i in order[:size]) == size // 2
    assert order == stratified_nested_order(rows, seed=1)
    assert order != stratified_nested_order(rows, seed=2)


def test_sets_have_required_sizes_and_nesting(tmp_path):
    raw = _toy(tmp_path)
    summary = _build(tmp_path, raw)
    sizes = {
        (k, p): e["n"] for k, parts in summary["conditions"].items() for p, e in parts.items()
    }
    assert sizes == {
        ("s2k", "train"): 2000,
        ("s5k", "train"): 5000,
        ("swap", "train"): 6040,
        ("tgt4k", "train"): 40,
        ("strict", "train"): 8000,
        ("strict", "val"): 2000,
    }
    ids = lambda name, part="train": {r["id"] for r in read_jsonl(raw / name / f"{part}.jsonl")}  # noqa: E731
    assert ids("s2k") < ids("s5k") < ids("strict")
    assert ids("s5k") < {i for i in ids("swap") if i.startswith("c1r")}
    assert ids("strict") .isdisjoint(ids("strict", "val"))
    assert all(i.startswith("c1r") for i in ids("strict", "val"))
    assert all(i.startswith("c2r") for i in ids("tgt4k"))
    assert summary["conditions"]["strict"]["val"]["label_counts"] == {"0": 1000, "1": 1000}


def test_generation_is_deterministic(tmp_path):
    raw = _toy(tmp_path)
    first = _build(tmp_path, raw)
    second = _build(tmp_path, raw)
    assert first == second
    other = _build(tmp_path, raw, seed=7)
    assert other["conditions"]["s2k"]["train"]["ids_sha256"] != first["conditions"]["s2k"]["train"]["ids_sha256"]


def test_test_set_overlap_is_fatal(tmp_path):
    raw = _toy(tmp_path)
    leaked = read_jsonl(raw / "eval_data.jsonl")
    leaked[0]["question"], leaked[0]["context"] = "question c1r5", "context c1r5"
    _write(raw / "eval_data.jsonl", leaked)
    with pytest.raises(RuntimeError, match="test"):
        _build(tmp_path, raw)


def test_train_validation_overlap_is_recorded_not_fatal(tmp_path):
    raw = _toy(tmp_path)
    val = read_jsonl(raw / "val.jsonl")
    val[0]["question"], val[0]["context"] = "question c2r3", "context c2r3"
    _write(raw / "val.jsonl", val)
    summary = _build(tmp_path, raw)
    assert summary["train_rows_overlapping_validation"] == {"swap": ["c2r3"], "tgt4k": ["c2r3"]}


def test_target_row_with_source_prefix_is_rejected(tmp_path):
    raw = _toy(tmp_path)
    rows = read_jsonl(raw / "train.jsonl")
    rows.append(_row("c1rEXTRA", 0))
    _write(raw / "train.jsonl", rows)
    with pytest.raises(ValueError, match="source"):
        _build(tmp_path, raw)


@pytest.mark.skipif(not (ROOT / "data/splits/splits_summary.json").is_file(), reason="splits not generated")
def test_shipped_splits_match_the_protocol():
    summary = json.loads((ROOT / "data/splits/splits_summary.json").read_text())
    c = summary["conditions"]
    assert summary["source_rows"] == 10000 and summary["target_rows"] == 4000
    assert c["s2k"]["train"]["n"] == 2000 and c["s5k"]["train"]["n"] == 5000
    assert c["swap"]["train"]["n_source"] == 6000 and c["swap"]["train"]["n_target"] == 4000
    assert c["tgt4k"]["train"]["n_source"] == 0
    assert c["strict"]["train"]["n"] == 8000 and c["strict"]["val"]["n"] == 2000
    assert c["strict"]["val"]["n_target"] == 0
    test = {json.loads(l)["question"] for l in (ROOT / "data/raw/eval_data.jsonl").open()}
    for name in ("s2k", "s5k", "swap", "tgt4k", "strict"):
        train = {json.loads(l)["question"] for l in (ROOT / f"data/raw/{name}/train.jsonl").open()}
        assert not train & test
