import csv
import json

import pytest

from clinhallu.data.external import build_unseen_set, convert_external


def _w(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_convert_filters_maps_labels_and_drops_overlap(tmp_path):
    rows = [
        {"question": "q1", "passage": "p1", "answer": "a1", "label": "PASS", "source_ds": "pubmedQA"},
        {"question": "q2", "passage": "p2", "answer": "a2", "label": "FAIL", "source_ds": "pubmedQA"},
        {"question": "q3", "passage": "p3", "answer": "a3", "label": "FAIL", "source_ds": "covidQA"},
        {"question": "q4", "passage": "p4", "answer": "a4", "label": "PASS", "source_ds": "pubmedQA"},
        {"question": "q1", "passage": "p1", "answer": "a1", "label": "PASS", "source_ds": "pubmedQA"},  # duplicate
    ]
    _w(tmp_path / "src.jsonl", rows)
    _w(tmp_path / "train.jsonl", [{"id": "t", "question": "q4", "context": "p4", "answer": "x", "hallu_label": 0}])
    card = convert_external(tmp_path / "src.jsonl", tmp_path / "ext.jsonl", exclude_files=[tmp_path / "train.jsonl"], audit_size=2)
    out = [json.loads(l) for l in (tmp_path / "ext.jsonl").open()]
    assert [(r["question"], r["hallu_label"]) for r in out] == [("q1", 0), ("q2", 1)]
    assert card["dropped_overlap_with_medhallu"] == 1 and card["dropped_exact_duplicates"] == 1
    assert card["label_counts"] == {"0": 1, "1": 1}
    assert len(list(csv.reader((tmp_path / "ext.audit.csv").open()))) == 3  # header + 2
    assert (tmp_path / "ext.card.json").is_file()
    with pytest.raises(ValueError, match="map label"):
        convert_external(_write_bad(tmp_path), tmp_path / "bad.jsonl")


def _write_bad(tmp_path):
    p = tmp_path / "bad.jsonl"
    _w(p, [{"question": "q", "passage": "p", "answer": "a", "label": "MAYBE", "source_ds": "pubmedQA"}])
    return p


class FakeCache:
    def __init__(self):
        self.calls = []

    def obtain(self, request, client):
        system = request["messages"][0]["content"]
        kind = "hallu" if "INCORRECT" not in system and "contradicted" in system else "supported"
        self.calls.append(kind)
        return {"answer": f"{kind} answer {len(self.calls)}"}, {}, False


def test_unseen_set_pairs_labels_and_skips_overlap(tmp_path):
    _w(tmp_path / "train.jsonl", [{"id": "t", "question": "q0", "context": "c0", "answer": "x", "hallu_label": 0}])
    pairs = [{"question": f"q{i}", "context": f"c{i}"} for i in range(4)]
    builder = lambda model, messages, schema, name, settings: {"model": model, "messages": messages}  # noqa: E731
    cache = FakeCache()
    card = build_unseen_set(pairs, client=None, cache=cache, model="vendor/other", out_path=tmp_path / "u.jsonl",
                            exclude_files=[tmp_path / "train.jsonl"], limit=2, audit_size=3, request_builder=builder)
    rows = [json.loads(l) for l in (tmp_path / "u.jsonl").open()]
    assert len(rows) == 4 and card["skipped_overlap_with_medhallu"] == 1
    assert [r["hallu_label"] for r in rows] == [0, 1, 0, 1]
    assert {r["question"] for r in rows} == {"q1", "q2"}
    assert all(r["generator"] == "vendor/other" for r in rows)
    assert cache.calls == ["supported", "hallu", "supported", "hallu"]
