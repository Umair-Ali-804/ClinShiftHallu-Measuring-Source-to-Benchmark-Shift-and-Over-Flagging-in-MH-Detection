"""External-benchmark and unseen-generator dataset builders (Phase 3 of the plan).

* ``convert_external``  : map a HaluBench-style table (PubMedQA slice) to the
  canonical ``question, context, answer, hallu_label`` JSONL, drop rows that overlap
  MedHallu train/val/test, and write a dataset card plus a 100-row manual-audit CSV.
* ``build_unseen_set``  : create a supported and a hallucinated answer for each
  held-out abstract with an LLM that is NOT the one that generated MedHallu.

Neither function retunes or trains anything; the resulting files are scored with
``clinhallu eval-only`` using a run's frozen checkpoint and calibration.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .io import read_jsonl, sha256_file, write_jsonl
from .leakage import canonical_group_key

HALUBENCH_LABELS = {"PASS": 0, "FAIL": 1}  # PASS = supported, FAIL = hallucinated


def _read_table(path: str | Path) -> list[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return read_jsonl(path)
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".parquet":
        import pandas as pd  # pyarrow required

        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _label(value, mapping: Mapping[str, int]) -> int:
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    key = str(value).strip()
    for candidate in (key, key.upper(), key.lower()):
        if candidate in mapping:
            return int(mapping[candidate])
    raise ValueError(f"Cannot map label {value!r}; pass a label_map covering it")


def _audit_csv(records: Sequence[Mapping], path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(records), size=min(n, len(records)), replace=False).tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "question", "context_first_600", "answer", "given_label", "reviewer_label", "notes"])
        for index in pick:
            r = records[index]
            writer.writerow([r["id"], r["question"], str(r["context"])[:600], r["answer"], r["hallu_label"], "", ""])


def convert_external(
    source: str | Path,
    out_path: str | Path,
    *,
    question_col: str = "question",
    context_col: str = "passage",
    answer_col: str = "answer",
    label_col: str = "label",
    filter_col: str | None = "source_ds",
    filter_value: str | None = "pubmedQA",
    label_map: Mapping[str, int] | None = None,
    exclude_files: Sequence[str | Path] = (),
    audit_size: int = 100,
    seed: int = 0,
    license_note: str = "unverified -- fill in from the dataset page before release",
) -> dict:
    rows = _read_table(source)
    if filter_col and filter_value is not None:
        rows = [r for r in rows if str(r.get(filter_col, "")).lower() == filter_value.lower()]
    if not rows:
        raise ValueError("No rows left after filtering; check filter_col/filter_value")
    mapping = dict(label_map or HALUBENCH_LABELS)
    excluded_groups: set[str] = set()
    for path in exclude_files:
        excluded_groups |= {canonical_group_key(r) for r in read_jsonl(path)}

    records, dropped_overlap, dropped_duplicate = [], 0, 0
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        record = {
            "id": f"ext{index}",
            "question": str(row[question_col]).strip(),
            "context": str(row[context_col]).strip(),
            "answer": str(row[answer_col]).strip(),
            "hallu_label": _label(row[label_col], mapping),
        }
        if not (record["question"] and record["context"] and record["answer"]):
            continue
        if canonical_group_key(record) in excluded_groups:
            dropped_overlap += 1
            continue
        key = (canonical_group_key(record), record["answer"])
        if key in seen:
            dropped_duplicate += 1
            continue
        seen.add(key)
        records.append(record)
    out_path = Path(out_path)
    write_jsonl(records, out_path)
    labels = Counter(r["hallu_label"] for r in records)
    card = {
        "name": out_path.stem,
        "source_file": str(source),
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(out_path),
        "n_rows": len(records),
        "label_counts": {str(k): v for k, v in sorted(labels.items())},
        "hallucinated_fraction": labels.get(1, 0) / max(1, len(records)),
        "label_definition": "1 = answer NOT supported by the passage (hallucinated); 0 = supported",
        "label_map": mapping,
        "filter": {filter_col: filter_value},
        "dropped_overlap_with_medhallu": dropped_overlap,
        "dropped_exact_duplicates": dropped_duplicate,
        "mean_answer_words": float(np.mean([len(r["answer"].split()) for r in records])),
        "license": license_note,
        "differences_from_medhallu": "Fill in after the manual audit: label definition, answer style/length, source journals.",
        "manual_audit": f"{out_path.with_suffix('.audit.csv').name}: fill reviewer_label, then report the disagreement rate",
    }
    out_path.with_suffix(".card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    _audit_csv(records, out_path.with_suffix(".audit.csv"), audit_size, seed)
    return card


# ----------------------------------------------------------------- unseen set
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
SUPPORTED_SYSTEM = (
    "You answer biomedical questions using ONLY the abstract provided. Write 2-4 sentences. "
    "Every claim must be directly stated in the abstract. Reply as JSON: {\"answer\": \"...\"}."
)
HALLUCINATED_SYSTEM = (
    "You write a fluent, confident answer to a biomedical question that LOOKS grounded in the "
    "abstract but contains at least one specific claim that is contradicted by, or absent from, "
    "the abstract (for example a reversed effect direction, a wrong number, or an invented "
    "mechanism). Keep the style and length of a normal 2-4 sentence answer. "
    "Reply as JSON: {\"answer\": \"...\"}."
)


def build_unseen_set(
    pairs: Iterable[Mapping],
    *,
    client,
    cache,
    model: str,
    out_path: str | Path,
    exclude_files: Sequence[str | Path] = (),
    limit: int = 500,
    audit_size: int = 50,
    settings: Mapping | None = None,
    seed: int = 0,
    request_builder: Callable | None = None,
) -> dict:
    """One supported + one hallucinated answer per held-out (question, context)."""
    if request_builder is None:
        from clinhallu.api.openrouter import build_request as request_builder
    settings = {"temperature": 0.7, "max_tokens": 400, **(settings or {})}
    excluded: set[str] = set()
    for path in exclude_files:
        excluded |= {canonical_group_key(r) for r in read_jsonl(path)}
    records, skipped = [], 0
    for index, pair in enumerate(pairs):
        if len(records) >= 2 * limit:
            break
        item = {"question": str(pair["question"]).strip(), "context": str(pair["context"]).strip()}
        if canonical_group_key(item) in excluded:
            skipped += 1
            continue
        for label, system in ((0, SUPPORTED_SYSTEM), (1, HALLUCINATED_SYSTEM)):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Abstract:\n{item['context']}\n\nQuestion: {item['question']}"},
            ]
            request = request_builder(model, messages, ANSWER_SCHEMA, "unseen_answer", settings)
            parsed, _response, _reused = cache.obtain(request, client)
            answer = str(parsed.get("answer", "")).strip()
            if not answer:
                raise ValueError(f"Empty answer for pair {index} label {label}")
            records.append(
                {
                    "id": f"unseen{index}_{label}",
                    "question": item["question"],
                    "context": item["context"],
                    "answer": answer,
                    "hallu_label": label,
                    "generator": model,
                }
            )
    if not records:
        raise ValueError("No records produced; every input pair overlapped the excluded splits")
    out_path = Path(out_path)
    write_jsonl(records, out_path)
    _audit_csv(records, out_path.with_suffix(".audit.csv"), audit_size, seed)
    card = {
        "name": out_path.stem,
        "generator": model,
        "n_rows": len(records),
        "label_counts": {"0": sum(r["hallu_label"] == 0 for r in records), "1": sum(r["hallu_label"] == 1 for r in records)},
        "skipped_overlap_with_medhallu": skipped,
        "label_note": "Labels are by construction (which prompt produced the answer) and are NOT human verified. "
        "Hand-check the audit CSV and report the error rate.",
        "settings": dict(settings),
        "output_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    out_path.with_suffix(".card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card
