"""Job list for the multi-account Colab study.

``build_jobs`` is the single source of truth: 33 training runs plus the
eval-only probes that depend on them.  ``est_minutes`` are placeholders -- replace
them with measured pilot times (``training_summary.json`` + log timestamps) and
re-run ``scripts/make_jobs.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

SEEDS = (13, 42, 77)

# condition -> training rows (for the time estimate only)
CONDITION_ROWS = {
    "c1": 10000, "c2": 11000, "c3": 12000, "c4": 13000, "c5": 14000,
    "swap": 10000, "tgt4k": 4000, "strict": 8000, "s2k": 2000, "s5k": 5000,
    "ansonly": 14000,
}

# plan section 4: balanced preferred lists per account
ACCOUNT_CONDITIONS = {
    "A1": ["c1", "c2"],
    "A2": ["c3", "c4"],
    "A3": ["c5", "swap"],
    "A4": ["strict", "tgt4k", "s2k"],
    "A5": ["s5k", "ansonly"],
}

RAW_TEST = "data/raw/eval_data.jsonl"
RAW_VAL = "data/raw/val.jsonl"
EXTERNAL = {"external": "data/external/halubench_pubmedqa.jsonl",
            "unseen": "data/external/unseen_generator.jsonl"}


def est_minutes(condition: str) -> int:
    return max(8, round(CONDITION_ROWS[condition] * 0.0026))


def run_id(condition: str, seed: int) -> str:
    return f"{condition}_seed{seed}"


def build_jobs() -> list[dict]:
    jobs: list[dict] = []
    for wave, seed in enumerate(SEEDS, start=1):
        for condition in CONDITION_ROWS:
            jobs.append(
                {
                    "run_id": run_id(condition, seed),
                    "kind": "train",
                    "condition": condition,
                    "config": f"configs/conditions/{condition}.yaml",
                    "seed": seed,
                    "est_minutes": est_minutes(condition),
                    "priority": wave,  # seed 13 for every condition first
                    "requires": [],
                }
            )

    def eval_job(condition: str, seed: int, tag: str, *, mode: str | None, dataset: str) -> dict:
        return {
            "run_id": f"eval_{condition}_seed{seed}_{tag}",
            "kind": "eval",
            "condition": condition,
            "config": f"configs/conditions/{condition}.yaml",
            "seed": seed,
            "tag": tag,
            "context_mode": mode,
            "dataset": dataset,
            "est_minutes": 6,
            "priority": SEEDS.index(seed) + 1.5,
            "requires": [run_id(condition, seed)],
        }

    for seed in SEEDS:
        for condition in ("c1", "c5"):
            jobs.append(eval_job(condition, seed, "blank", mode="blank", dataset=RAW_TEST))
            jobs.append(eval_job(condition, seed, "shuffled", mode="shuffled", dataset=RAW_TEST))
            for tag, path in EXTERNAL.items():
                jobs.append(eval_job(condition, seed, tag, mode=None, dataset=path))
        # STRICT scored on TARGET validation: lets analysis refit T/tau on target labels
        jobs.append(eval_job("strict", seed, "targetval", mode=None, dataset=RAW_VAL))
    ids = [j["run_id"] for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate run ids"
    return jobs


def preferred_lists(jobs: list[dict]) -> dict[str, list[str]]:
    lists: dict[str, list[str]] = {}
    for account, conditions in ACCOUNT_CONDITIONS.items():
        mine = [j for j in jobs if j["condition"] in conditions]
        mine.sort(key=lambda j: (j["priority"], j["kind"] != "train", j["run_id"]))
        lists[account] = [j["run_id"] for j in mine]
    return lists


def write_jobs(directory: str | Path) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "claims").mkdir(exist_ok=True)
    jobs = build_jobs()
    (directory / "jobs.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
    lists = preferred_lists(jobs)
    for account, ids in lists.items():
        (directory / f"jobs_{account}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    return {
        "jobs": len(jobs),
        "training_runs": sum(j["kind"] == "train" for j in jobs),
        "eval_jobs": sum(j["kind"] == "eval" for j in jobs),
    }
