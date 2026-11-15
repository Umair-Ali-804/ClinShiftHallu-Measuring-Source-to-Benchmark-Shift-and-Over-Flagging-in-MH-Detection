

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

EXCLUDE_TOP = {"artifacts", "jobs", ".git", "__pycache__", ".pytest_cache", ".venv"}
RUN_SUBDIR = "artifacts/adaptation"


# ------------------------------------------------------------------ file utils
def _same(src: Path, dst: Path) -> bool:
    try:
        a, b = src.stat(), dst.stat()
    except FileNotFoundError:
        return False
    return a.st_size == b.st_size and int(a.st_mtime) <= int(b.st_mtime) + 2


def mirror_tree(src: Path, dst: Path, *, skip_top: Sequence[str] = ()) -> int:
    """Copy new/changed files from ``src`` to ``dst``. Returns files copied."""
    copied = 0
    if not src.exists():
        return 0
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if rel.parts and rel.parts[0] in skip_top:
            continue
        if any(part == "__pycache__" for part in rel.parts) or path.suffix == ".tmp":
            continue
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file() and not _same(path, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            shutil.copy2(path, tmp)
            os.replace(tmp, target)
            copied += 1
    return copied


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------- jobs
@dataclass
class Job:
    raw: dict

    @property
    def run_id(self) -> str:
        return self.raw["run_id"]

    @property
    def kind(self) -> str:
        return self.raw["kind"]

    @property
    def rel_run_dir(self) -> str:
        return f"{RUN_SUBDIR}/{self.raw['condition']}/seed_{self.raw['seed']}"

    @property
    def done_name(self) -> str:
        return "DONE" if self.kind == "train" else f"eval_{self.raw['tag']}.DONE"

    @property
    def metrics_rel(self) -> str:
        name = "evaluation_metrics.json" if self.kind == "train" else f"{self.raw['tag']}_metrics.json"
        return f"{self.rel_run_dir}/metrics/{name}"


class Worker:
    def __init__(
        self,
        *,
        drive_root: Path,
        local_root: Path,
        account: str,
        jobs: list[dict],
        preferred: Sequence[str] = (),
        max_hours: float = 10.0,
        stale_hours: float = 3.0,
        sync_minutes: float = 10.0,
        runner: Optional[Callable[[Job, Path], int]] = None,
        clock: Callable[[], float] = time.time,
        only_preferred: bool = False,
    ) -> None:
        self.drive, self.local, self.account = Path(drive_root), Path(local_root), account
        self.jobs = [Job(j) for j in jobs]
        self.by_id = {j.run_id: j for j in self.jobs}
        self.max_seconds = max_hours * 3600
        self.stale_seconds = stale_hours * 3600
        self.sync_seconds = sync_minutes * 60
        self.runner = runner or self._default_runner
        self.clock = clock
        self.started = clock()
        self.failed: set[str] = set()
        order = {rid: i for i, rid in enumerate(preferred)}
        unknown = [rid for rid in preferred if rid not in self.by_id]
        if unknown:
            raise ValueError(f"Preferred list names unknown jobs: {unknown[:5]}")
        pool = [j for j in self.jobs if j.run_id in order] if only_preferred else self.jobs
        self.queue = sorted(
            pool,
            key=lambda j: (
                0 if j.run_id in order else 1,
                order.get(j.run_id, 0),
                j.raw.get("priority", 99),
                j.run_id,
            ),
        )

    # ------------------------------------------------------------- bookkeeping
    def drive_dir(self, job: Job) -> Path:
        return self.drive / job.rel_run_dir

    def local_dir(self, job: Job) -> Path:
        return self.local / job.rel_run_dir

    def is_done(self, job: Job) -> bool:
        return (self.drive_dir(job) / job.done_name).is_file()

    def claim_path(self, job: Job) -> Path:
        return self.drive / "jobs" / "claims" / f"{job.run_id}.claim"

    def read_claim(self, job: Job) -> Optional[dict]:
        try:
            return json.loads(self.claim_path(job).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def try_claim(self, job: Job) -> bool:
        path = self.claim_path(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"account": self.account, "timestamp": self.clock(), "run_id": job.run_id}
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = self.read_claim(job)
            fresh = existing and self.clock() - float(existing.get("timestamp", 0)) < self.stale_seconds
            if fresh and existing.get("account") != self.account:
                return False
            atomic_write_json(path, payload)  # stale takeover, or our own earlier claim
            return True
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return True

    def still_mine(self, job: Job) -> bool:
        claim = self.read_claim(job)
        return bool(claim and claim.get("account") == self.account)

    def heartbeat(self, job: Job) -> bool:
        if not self.still_mine(job):
            return False
        atomic_write_json(
            self.claim_path(job),
            {"account": self.account, "timestamp": self.clock(), "run_id": job.run_id},
        )
        return True

    def release(self, job: Job) -> None:
        if self.still_mine(job):
            try:
                self.claim_path(job).unlink()
            except FileNotFoundError:
                pass

    # ----------------------------------------------------------------- project
    def prepare_local_project(self, refresh: bool = False) -> None:
        self.local.mkdir(parents=True, exist_ok=True)
        if refresh or not (self.local / "pyproject.toml").is_file():
            for item in self.drive.iterdir():
                if item.name in EXCLUDE_TOP:
                    continue
                if item.is_dir():
                    mirror_tree(item, self.local / item.name)
                elif item.is_file():
                    shutil.copy2(item, self.local / item.name)
        (self.local / "artifacts").mkdir(exist_ok=True)

    # ------------------------------------------------------------------ running
    def _default_runner(self, job: Job, cwd: Path) -> int:
        raw = job.raw
        base = [sys.executable, "-m", "clinhallu"]
        if job.kind == "train":
            cmd = base + ["run", "--config", raw["config"], "--seed", str(raw["seed"])]
        else:
            cmd = base + [
                "eval-only", "--config", raw["config"], "--seed", str(raw["seed"]),
                "--tag", raw["tag"], "--dataset", raw["dataset"],
            ]
            if raw.get("context_mode"):
                cmd += ["--context-mode", raw["context_mode"]]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(cwd / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["CLINHALLU_ROOT"] = str(cwd)
        return subprocess.call(cmd, cwd=cwd, env=env)

    def _runnable(self, job: Job) -> tuple[bool, str]:
        if self.is_done(job):
            return False, "done"
        if job.run_id in self.failed:
            return False, "failed-this-session"
        for need in job.raw.get("requires", []):
            if need not in self.by_id or not self.is_done(self.by_id[need]):
                return False, f"waiting-for:{need}"
        if job.kind == "eval":
            dataset = self.local / job.raw["dataset"]
            if not dataset.is_file() and not (self.drive / job.raw["dataset"]).is_file():
                return False, f"missing-dataset:{job.raw['dataset']}"
        elapsed = self.clock() - self.started
        if elapsed + job.raw.get("est_minutes", 30) * 60 * 1.25 > self.max_seconds:
            return False, "not-enough-time"
        claim = self.read_claim(job)
        if claim and claim.get("account") != self.account:
            if self.clock() - float(claim.get("timestamp", 0)) < self.stale_seconds:
                return False, f"claimed-by:{claim.get('account')}"
        return True, "ok"

    def run_job(self, job: Job) -> bool:
        if not self.try_claim(job):
            return False
        local, drive = self.local_dir(job), self.drive_dir(job)
        # resume point: whatever the last owner mirrored to Drive
        resumed = drive.exists() and any(
            path.name.startswith("latest_checkpoint") for path in drive.rglob("*")
        )
        mirror_tree(drive, local)
        if job.kind == "eval":
            dataset = self.local / job.raw["dataset"]
            if not dataset.is_file():
                mirror_tree(self.drive / Path(job.raw["dataset"]).parent, dataset.parent)
        stop = threading.Event()
        lost = threading.Event()

        def pump() -> None:
            while not stop.wait(self.sync_seconds):
                if not self.heartbeat(job):
                    lost.set()
                    return
                mirror_tree(local, drive)

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        code = -1
        started = self.clock()
        try:
            code = self.runner(job, self.local)
        finally:
            stop.set()
            thread.join(timeout=5)
            if not lost.is_set():
                mirror_tree(local, drive)
        if lost.is_set():
            print(f"[{job.run_id}] claim was taken over by another account; leaving it to them")
            return False
        ok = code == 0 and (self.local / job.metrics_rel).is_file()
        if ok:
            atomic_write_json(
                drive / job.done_name,
                {
                    "account": self.account,
                    "finished": self.clock(),
                    "run_id": job.run_id,
                    "kind": job.kind,
                    "condition": job.raw.get("condition"),
                    "seed": job.raw.get("seed"),
                    # wall time of THIS session only; a resumed run under-reports.
                    "duration_minutes": round((self.clock() - started) / 60.0, 2),
                    "resumed": bool(resumed),
                },
            )
            self.release(job)
            print(f"[{job.run_id}] DONE")
            return True
        self.failed.add(job.run_id)
        self.release(job)
        reason = "interrupted (resumable)" if code == 130 else f"exit code {code}"
        print(f"[{job.run_id}] not completed: {reason}")
        return False

    def run(self, poll_minutes: float = 0.0, dry_run: bool = False) -> dict:
        self.prepare_local_project()
        summary = {"completed": [], "failed": [], "skipped": {}}
        while True:
            progressed = False
            blocked: dict[str, str] = {}
            for job in self.queue:
                ok, reason = self._runnable(job)
                if not ok:
                    if reason != "done":
                        blocked[job.run_id] = reason
                    continue
                if dry_run:
                    print(f"would run {job.run_id} ({job.kind}, ~{job.raw.get('est_minutes')} min)")
                    summary["completed"].append(job.run_id)
                    continue
                progressed = True
                if self.run_job(job):
                    summary["completed"].append(job.run_id)
                else:
                    summary["failed"].append(job.run_id)
            transient = {k: v for k, v in blocked.items() if v.startswith(("waiting", "claimed"))}
            if dry_run or progressed:
                if dry_run:
                    summary["skipped"] = blocked
                    return summary
                continue
            summary["skipped"] = blocked
            elapsed = self.clock() - self.started
            if poll_minutes > 0 and transient and elapsed + poll_minutes * 60 < self.max_seconds:
                time.sleep(poll_minutes * 60)
                continue
            return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", required=True, type=Path)
    parser.add_argument("--local-root", default=Path("/content/work/ClinHallu"), type=Path)
    parser.add_argument("--account", required=True, help="A1..A5")
    parser.add_argument("--preferred", help="jobs_<account>.txt (default: jobs/jobs_<account>.txt)")
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--stale-hours", type=float, default=3.0)
    parser.add_argument("--sync-minutes", type=float, default=10.0)
    parser.add_argument("--poll-minutes", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only-preferred",
        action="store_true",
        help="Run ONLY the jobs in the preferred list (e.g. the Day 7 pilot list)",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        metavar="RUN_ID",
        help="Run ONLY these job ids (e.g. c1_seed13), nothing else. One notebook cell = one run.",
    )
    args = parser.parse_args(argv)
    jobs = json.loads((args.drive_root / "jobs" / "jobs.json").read_text(encoding="utf-8"))
    pref_path = Path(args.preferred) if args.preferred else args.drive_root / "jobs" / f"jobs_{args.account}.txt"
    preferred = pref_path.read_text(encoding="utf-8").split() if pref_path.is_file() else []
    if args.run:
        preferred = list(args.run)
        args.only_preferred = True
    worker = Worker(
        drive_root=args.drive_root,
        local_root=args.local_root,
        account=args.account,
        jobs=jobs,
        preferred=preferred,
        max_hours=args.max_hours,
        stale_hours=args.stale_hours,
        sync_minutes=args.sync_minutes,
        only_preferred=args.only_preferred,
    )
    print(json.dumps(worker.run(poll_minutes=args.poll_minutes, dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
