import json
from pathlib import Path

from clinhallu.colab.jobs import build_jobs, preferred_lists, write_jobs
from clinhallu.colab.worker import Worker, mirror_tree


def _project(tmp_path):
    drive = tmp_path / "drive"
    (drive / "src").mkdir(parents=True)
    (drive / "pyproject.toml").write_text("x")
    (drive / "src" / "m.py").write_text("print(1)")
    (drive / "data" / "raw").mkdir(parents=True)
    (drive / "data" / "raw" / "eval_data.jsonl").write_text("{}\n")
    return drive, tmp_path / "local"


def _jobs():
    return [
        {"run_id": "a_seed1", "kind": "train", "condition": "a", "seed": 1, "config": "c.yaml", "est_minutes": 10, "priority": 1, "requires": []},
        {"run_id": "b_seed1", "kind": "train", "condition": "b", "seed": 1, "config": "c.yaml", "est_minutes": 10, "priority": 1, "requires": []},
        {"run_id": "eval_a_blank", "kind": "eval", "condition": "a", "seed": 1, "config": "c.yaml", "tag": "blank",
         "context_mode": "blank", "dataset": "data/raw/eval_data.jsonl", "est_minutes": 2, "priority": 1.5, "requires": ["a_seed1"]},
    ]


def _fake_runner(log, fail=()):
    def run(job, cwd):
        log.append(job.run_id)
        if job.run_id in fail:
            return 1
        metrics = cwd / job.metrics_rel
        metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_text("{}")
        (cwd / job.rel_run_dir / f"out_{job.run_id}").write_text(job.run_id)
        return 0
    return run


def _worker(tmp_path, account="A1", clock=None, **kw):
    drive, local = _project(tmp_path) if not (tmp_path / "drive").exists() else (tmp_path / "drive", tmp_path / f"local_{account}")
    log = kw.pop("log", [])
    jobs = kw.pop("jobs", _jobs())
    args = dict(drive_root=drive, local_root=local, account=account, jobs=jobs, runner=_fake_runner(log, kw.pop("fail", ())), sync_minutes=60)
    if clock:
        args["clock"] = clock
    args.update(kw)
    return Worker(**args), log, drive


def test_runs_jobs_in_dependency_order_and_marks_done(tmp_path):
    worker, log, drive = _worker(tmp_path, preferred=["b_seed1"])
    summary = worker.run()
    assert log == ["b_seed1", "a_seed1", "eval_a_blank"]
    assert summary["failed"] == []
    for rel in ("artifacts/adaptation/a/seed_1/DONE", "artifacts/adaptation/b/seed_1/DONE", "artifacts/adaptation/a/seed_1/eval_blank.DONE"):
        assert (drive / rel).is_file()
    assert (drive / "artifacts/adaptation/a/seed_1/out_a_seed1").read_text() == "a_seed1"
    assert (drive / "artifacts/adaptation/a/seed_1/out_eval_a_blank").is_file()
    assert not list((drive / "jobs" / "claims").glob("*.claim"))  # all released
    # a second worker has nothing left to do
    other, log2, _ = _worker(tmp_path, account="A2")
    assert other.run()["completed"] == [] and log2 == []


def test_fresh_claim_blocks_other_accounts_and_stale_claim_is_taken_over(tmp_path):
    now = [1000.0]
    w1, log1, drive = _worker(tmp_path, "A1", clock=lambda: now[0])
    job = w1.by_id["a_seed1"]
    assert w1.try_claim(job)
    w2, log2, _ = _worker(tmp_path, "A2", clock=lambda: now[0])
    w2.run()
    assert "a_seed1" not in log2 and "b_seed1" in log2  # skipped a's, did b
    now[0] += 3 * 3600 + 1  # claim goes stale
    w3, log3, _ = _worker(tmp_path, "A3", clock=lambda: now[0])
    w3.run()
    assert "a_seed1" in log3


def test_failed_job_is_not_marked_done_and_not_retried_this_session(tmp_path):
    worker, log, drive = _worker(tmp_path, fail={"a_seed1"})
    summary = worker.run()
    assert summary["failed"] == ["a_seed1"]
    assert log.count("a_seed1") == 1
    assert not (drive / "artifacts/adaptation/a/seed_1/DONE").exists()
    assert "eval_a_blank" not in log  # dependency never completed
    assert not (drive / "jobs/claims/a_seed1.claim").exists()  # released for retry elsewhere


def test_resume_copies_previous_owner_state_from_drive(tmp_path):
    worker, log, drive = _worker(tmp_path)
    partial = drive / "artifacts/adaptation/a/seed_1/ckpt_latest"
    partial.parent.mkdir(parents=True)
    partial.write_text("progress")
    seen = []
    def runner(job, cwd):
        if job.run_id == "a_seed1":
            seen.append((cwd / job.rel_run_dir / "ckpt_latest").read_text())
        return _fake_runner([])(job, cwd)
    worker.runner = runner
    worker.run()
    assert seen[0] == "progress"


def test_time_budget_and_missing_dataset(tmp_path):
    t = [0.0]
    worker, log, _ = _worker(tmp_path, clock=lambda: t[0], max_hours=0.1)  # 6 min, jobs need 12.5
    worker.run()
    assert log == []
    jobs = _jobs()
    jobs[2]["dataset"] = "data/external/nope.jsonl"
    worker, log, _ = _worker(tmp_path, "A4", jobs=jobs)
    summary = worker.run()
    assert "eval_a_blank" not in log
    assert summary["skipped"]["eval_a_blank"].startswith("missing-dataset")


def test_mirror_is_incremental(tmp_path):
    src, dst = tmp_path / "s", tmp_path / "d"
    (src / "x").mkdir(parents=True)
    (src / "x" / "f").write_text("1")
    assert mirror_tree(src, dst) == 1
    assert mirror_tree(src, dst) == 0
    (src / "x" / "f").write_text("22")
    assert mirror_tree(src, dst) == 1


def test_job_list_matches_the_plan(tmp_path):
    jobs = build_jobs()
    train = [j for j in jobs if j["kind"] == "train"]
    assert len(train) == 33 and {j["seed"] for j in train} == {13, 42, 77}
    assert {j["condition"] for j in train} == {"c1", "c2", "c3", "c4", "c5", "swap", "tgt4k", "strict", "s2k", "s5k", "ansonly"}
    ids = {j["run_id"] for j in jobs}
    for j in jobs:
        assert all(r in ids for r in j["requires"])
    lists = preferred_lists(jobs)
    assigned = [rid for ids_ in lists.values() for rid in ids_]
    assert sorted(assigned) == sorted(ids)  # every job on exactly one preferred list
    first = [r for r in lists["A1"] if r.startswith(("c1_", "c2_"))][:2]
    assert all("seed13" in r for r in first)
    info = write_jobs(tmp_path / "jobs")
    assert info["training_runs"] == 33 and (tmp_path / "jobs" / "jobs_A5.txt").is_file()
    assert json.loads((tmp_path / "jobs" / "jobs.json").read_text())[0]["run_id"]
