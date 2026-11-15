from __future__ import annotations

from collections import Counter
from pathlib import Path

from .io import sha256_file
from .preprocessing import assert_group_disjoint, cross_split_overlap_report, load_input


def validate_fixed_splits(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    *,
    expected_rows: dict[str, int] | None = None,
    minimum_class_fraction: float = 0.0,
    overlap_policy: str = "error",
) -> dict:
    """Validate the three fixed splits.

    ``overlap_policy='error'`` rejects any question/context group shared by two
    splits.  ``'remove_lower_priority'`` tolerates overlap that involves the
    *training* split only (preprocessing then drops those training rows and
    records them in ``removed_cross_split_overlap.jsonl``, priority
    test > validation > train).  Overlap between validation and test is always
    an error: evaluation rows are never removed.
    """
    paths = {
        "train": Path(train_path),
        "validation": Path(validation_path),
        "test": Path(test_path),
    }
    records = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} split: {path}")
        records[name] = load_input(str(path))
        if not records[name]:
            raise ValueError(f"{name} split is empty: {path}")

    expected_rows = expected_rows or {}
    for name, expected in expected_rows.items():
        if expected is not None and len(records[name]) != int(expected):
            raise ValueError(
                f"{name} row-count mismatch: expected {expected}, observed {len(records[name])}"
            )

    policy = str(overlap_policy).lower()
    overlap_report = cross_split_overlap_report(*records.items())
    if policy == "remove_lower_priority":
        if overlap_report["validation__test"]["count"]:
            raise RuntimeError(
                "Data leakage: validation and test share "
                f"{overlap_report['validation__test']['count']} question/context group(s); "
                "evaluation rows are never removed automatically."
            )
    elif policy in {"error", "warn", "ignore"}:
        if policy == "error":
            assert_group_disjoint(*records.items())
    else:
        raise ValueError(f"Unknown overlap_policy {overlap_policy!r}")
    label_counts = {}
    for name, rows in records.items():
        counts = Counter(int(row["hall_label"]) for row in rows)
        if set(counts) != {0, 1}:
            raise ValueError(f"{name} must contain both labels; observed {dict(counts)}")
        minority = min(counts.values()) / len(rows)
        if minority < minimum_class_fraction:
            raise ValueError(
                f"{name} minority fraction {minority:.3f} is below {minimum_class_fraction:.3f}"
            )
        label_counts[name] = {str(k): int(v) for k, v in sorted(counts.items())}

    return {
        "counts": {name: len(rows) for name, rows in records.items()},
        "label_counts": label_counts,
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "overlap_policy": policy,
        "overlap_group_counts": {
            pair: int(item["count"]) for pair, item in overlap_report.items()
        },
    }
