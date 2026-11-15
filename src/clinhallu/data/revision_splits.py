"""Training-set constructions for the size-vs-composition revision study.

The original release ships C1-C5 as cumulative supersets.  The revision needs
extra training sets built from the *same* rows, so every subset here is derived
by row ID (never by slicing a file) and written together with an ID manifest.

Definitions (see docs/revision_protocol.md):

    source rows  = the 10,000 C1 rows (IDs c1r*), PubMedQA-derived
    target rows  = the 4,000 rows that C5 adds on top of C1 (IDs c2r*-c5r*)

    S2K     2,000 source                     nested  S2K < S5K < SWAP-src < STRICT-train < C1
    S5K     5,000 source
    SWAP    6,000 source + all 4,000 target  (same total size as C1)
    TGT4K   4,000 target only
    STRICT  8,000 source train / 2,000 source holdout used for checkpoint
            selection, temperature and threshold (no target label is used)

Nesting is obtained from ONE seeded, label-interleaved ordering of the source
rows, so every even-length prefix is exactly class balanced.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .io import read_jsonl, sha256_file, write_jsonl
from .leakage import canonical_group_key

DEFAULT_SEED = 2026
SOURCE_ID_PREFIX = "c1r"

SOURCE_PREFIX_SIZES = {
    "s2k": 2000,
    "s5k": 5000,
    "swap_source": 6000,
    "strict_train": 8000,
}


def _label(row: Mapping) -> int:
    return int(row["hallu_label"])


def stratified_nested_order(rows: Sequence[Mapping], seed: int = DEFAULT_SEED) -> list[str]:
    """Return source IDs ordered so that every prefix is (near) class balanced."""
    positives = sorted(str(r["id"]) for r in rows if _label(r) == 1)
    negatives = sorted(str(r["id"]) for r in rows if _label(r) == 0)
    rng = np.random.default_rng(seed)
    positives = [positives[i] for i in rng.permutation(len(positives))]
    negatives = [negatives[i] for i in rng.permutation(len(negatives))]
    order: list[str] = []
    for index in range(max(len(positives), len(negatives))):
        if index < len(positives):
            order.append(positives[index])
        if index < len(negatives):
            order.append(negatives[index])
    return order


def _subset(rows: Sequence[Mapping], ids: Iterable[str]) -> list[dict]:
    wanted = set(ids)
    return [dict(r) for r in rows if str(r["id"]) in wanted]


def _ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _label_counts(rows: Sequence[Mapping]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(_label(r) for r in rows).items())}


def _group_keys(rows: Sequence[Mapping]) -> set[str]:
    return {canonical_group_key(r) for r in rows}


def build_revision_splits(
    *,
    c1_path: str | Path,
    c5_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    raw_dir: str | Path,
    splits_dir: str | Path,
    seed: int = DEFAULT_SEED,
    source_prefix: str = SOURCE_ID_PREFIX,
) -> dict:
    c1 = read_jsonl(c1_path)
    c5 = read_jsonl(c5_path)
    val = read_jsonl(val_path)
    test = read_jsonl(test_path)

    c1_ids = [str(r["id"]) for r in c1]
    if len(set(c1_ids)) != len(c1_ids):
        raise ValueError("C1 contains duplicate IDs")
    bad = [i for i in c1_ids if not i.startswith(source_prefix)]
    if bad:
        raise ValueError(f"{len(bad)} C1 IDs do not start with {source_prefix!r}, e.g. {bad[0]}")
    c5_ids = {str(r["id"]) for r in c5}
    missing = set(c1_ids) - c5_ids
    if missing:
        raise ValueError(f"C1 is not a subset of C5: {len(missing)} C1 IDs absent from C5")

    source = c1
    target = [r for r in c5 if str(r["id"]) not in set(c1_ids)]
    if any(str(r["id"]).startswith(source_prefix) for r in target):
        raise ValueError("A C5-only row carries a source ID prefix; source/target split is unsafe")
    if len(source) + len(target) != len(c5):
        raise ValueError("source + target rows do not add up to C5")

    order = stratified_nested_order(source, seed)
    prefix_ids = {name: order[:size] for name, size in SOURCE_PREFIX_SIZES.items()}
    holdout_ids = order[SOURCE_PREFIX_SIZES["strict_train"] :]

    sets = {
        "s2k": {"train": _subset(source, prefix_ids["s2k"])},
        "s5k": {"train": _subset(source, prefix_ids["s5k"])},
        "swap": {"train": _subset(source, prefix_ids["swap_source"]) + [dict(r) for r in target]},
        "tgt4k": {"train": [dict(r) for r in target]},
        "strict": {
            "train": _subset(source, prefix_ids["strict_train"]),
            "val": _subset(source, holdout_ids),
        },
    }

    # ---- audits: nothing here may leak into the frozen test split ----------
    test_groups = _group_keys(test)
    val_groups = _group_keys(val)
    overlap_with_val: dict[str, list[str]] = {}
    for name, parts in sets.items():
        for part, rows in parts.items():
            if len(rows) != len({str(r["id"]) for r in rows}):
                raise RuntimeError(f"{name}/{part}: duplicate IDs")
            groups = _group_keys(rows)
            if groups & test_groups:
                raise RuntimeError(f"{name}/{part}: question/context overlap with the test split")
            if part == "train":
                shared = groups & val_groups
                if shared:
                    # Not fatal: preprocessing (overlap_policy=remove_lower_priority)
                    # drops these TRAIN rows; recorded here for the audit trail.
                    overlap_with_val[name] = sorted(
                        str(r["id"]) for r in rows if canonical_group_key(r) in shared
                    )
    strict_train_groups = _group_keys(sets["strict"]["train"])
    strict_val_groups = _group_keys(sets["strict"]["val"])
    if strict_train_groups & strict_val_groups:
        raise RuntimeError("STRICT train and source holdout share question/context groups")
    if {str(r["id"]) for r in sets["strict"]["train"]} & {
        str(r["id"]) for r in sets["strict"]["val"]
    }:
        raise RuntimeError("STRICT train and holdout share IDs")
    # nesting
    chain = ["s2k", "s5k", "swap_source", "strict_train"]
    for small, large in zip(chain, chain[1:]):
        if not set(prefix_ids[small]) <= set(prefix_ids[large]):
            raise RuntimeError(f"nesting violated: {small} not inside {large}")

    raw_dir = Path(raw_dir)
    splits_dir = Path(splits_dir)
    summary: dict = {
        "seed": seed,
        "source_id_prefix": source_prefix,
        "inputs_sha256": {
            "c1": sha256_file(c1_path),
            "c5": sha256_file(c5_path),
            "val": sha256_file(val_path),
            "test": sha256_file(test_path),
        },
        "source_rows": len(source),
        "target_rows": len(target),
        "target_label_counts": _label_counts(target),
        "train_rows_overlapping_validation": overlap_with_val,
        "conditions": {},
    }
    for name, parts in sets.items():
        entry = {}
        for part, rows in parts.items():
            filename = "train.jsonl" if part == "train" else "val.jsonl"
            destination = raw_dir / name / filename
            write_jsonl(rows, destination)
            ids = [str(r["id"]) for r in rows]
            manifest = {
                "condition": name,
                "part": part,
                "n": len(rows),
                "label_counts": _label_counts(rows),
                "n_source": sum(i.startswith(source_prefix) for i in ids),
                "n_target": sum(not i.startswith(source_prefix) for i in ids),
                "ids_sha256": _ids_digest(ids),
                "file": str(destination),
                "file_sha256": sha256_file(destination),
                "ids": sorted(ids),
            }
            splits_dir.mkdir(parents=True, exist_ok=True)
            (splits_dir / f"{name}_{part}_ids.json").write_text(
                json.dumps(manifest, indent=1), encoding="utf-8"
            )
            entry[part] = {k: v for k, v in manifest.items() if k != "ids"}
        summary["conditions"][name] = entry
    (splits_dir / "splits_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
