from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_predictions(
    samples: Iterable[dict],
    labels,
    raw_probabilities,
    calibrated_probabilities,
    predictions,
    path: str | Path,
    evidence_stats: list[dict] | None = None,
    *,
    logits=None,
    metadata: list[dict] | None = None,
) -> None:
    """Write one JSON object per example.

    ``logits`` (raw, pre-temperature decision scores) allows temperature, Platt and
    threshold refits from the file alone.  ``metadata`` carries per-example
    descriptors from ``example_metadata``.
    """
    samples = list(samples)
    labels = list(labels)
    raw_probabilities = list(raw_probabilities)
    calibrated_probabilities = list(calibrated_probabilities)
    predictions = list(predictions)
    expected = len(samples)
    lengths = {
        "labels": len(labels),
        "raw_probabilities": len(raw_probabilities),
        "calibrated_probabilities": len(calibrated_probabilities),
        "predictions": len(predictions),
    }
    if any(length != expected for length in lengths.values()):
        raise ValueError(f"Prediction output is misaligned: samples={expected}, {lengths}")

    logits = list(logits) if logits is not None else None
    if logits is not None and len(logits) != expected:
        raise ValueError(f"Logits are misaligned: expected {expected}, got {len(logits)}")
    metadata = list(metadata) if metadata is not None else None
    if metadata is not None and len(metadata) != expected:
        raise ValueError(f"Metadata is misaligned: expected {expected}, got {len(metadata)}")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stats = evidence_stats or [{} for _ in predictions]
    if len(stats) != expected:
        raise ValueError(
            f"Evidence statistics are misaligned: expected {expected}, got {len(stats)}"
        )
    with destination.open("w", encoding="utf-8") as handle:
        for position, (sample, label, raw, calibrated, prediction, extra) in enumerate(
            zip(
                samples,
                labels,
                raw_probabilities,
                calibrated_probabilities,
                predictions,
                stats,
                strict=True,
            )
        ):
            record = {
                "id": sample.get("sample_id", sample.get("id")),
                "hallu_label": int(label),
                "prediction": int(prediction),
                "p_hallucinated_raw": float(raw),
                "p_hallucinated_calibrated": float(calibrated),
            }
            if logits is not None:
                record["logit"] = float(logits[position])
            if metadata is not None:
                record.update(metadata[position])
            record.update(extra)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
