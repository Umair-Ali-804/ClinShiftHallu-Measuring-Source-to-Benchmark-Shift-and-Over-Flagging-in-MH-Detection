import json

import pytest

pytest.importorskip("torch")

from clinhallu.data.cache import cache_key, _cache_identity  # noqa: E402


class Tok:
    name_or_path, vocab_size, cls_token_id, sep_token_id, pad_token_id = "tok", 10, 1, 2, 0


def _cfg(tmp_path, mode="normal", seed=20260):
    for name in ("train", "validation", "test"):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps({"x": name}) + "\n")
    return {
        "data": {"processed_dir": str(tmp_path), "question_budget": 96, "answer_budget": 128,
                 "context_mode": mode, "context_shuffle_seed": seed},
        "model": {"max_length": 512, "revision": "r"},
    }


def test_context_mode_changes_the_cache_directory_key(tmp_path):
    keys = {mode: cache_key(_cache_identity(_cfg(tmp_path, mode), Tok())) for mode in ("normal", "blank", "shuffled")}
    assert len(set(keys.values())) == 3


def test_shuffle_seed_matters_only_when_shuffling(tmp_path):
    key = lambda mode, seed: cache_key(_cache_identity(_cfg(tmp_path, mode, seed), Tok()))  # noqa: E731
    assert key("shuffled", 1) != key("shuffled", 2)
    assert key("blank", 1) == key("blank", 2)
