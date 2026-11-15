import pytest

from clinhallu.data.context_modes import apply_context_mode, derange_contexts


def _samples(n, duplicate_every=0):
    rows = []
    for i in range(n):
        ctx = f"ctx {i // duplicate_every}" if duplicate_every else f"ctx {i}"
        rows.append({"sample_id": f"s{i}", "question": f"q{i}", "context": ctx, "answer": f"a{i}", "hall_label": i % 2})
    return rows


def test_derangement_is_a_permutation_with_no_fixed_text():
    contexts = [f"c{i}" for i in range(200)]
    perm = derange_contexts(contexts, seed=3)
    assert sorted(perm) == list(range(200))
    assert all(contexts[perm[i]] != contexts[i] for i in range(200))
    assert perm == derange_contexts(contexts, seed=3)
    assert perm != derange_contexts(contexts, seed=4)


def test_duplicate_contexts_never_stay_with_their_owner():
    samples = _samples(120, duplicate_every=3)  # each context appears 3 times
    out = apply_context_mode(samples, "shuffled", seed=11)
    for before, after in zip(samples, out):
        assert before["context"] != after["context"]


def test_shuffle_changes_only_context_and_preserves_the_context_multiset():
    samples = _samples(50)
    out = apply_context_mode(samples, "shuffled", seed=0)
    for before, after in zip(samples, out):
        for key in ("sample_id", "question", "answer", "hall_label"):
            assert before[key] == after[key]
        assert after["context_mode"] == "shuffled"
    assert sorted(s["context"] for s in out) == sorted(s["context"] for s in samples)
    assert samples[0]["context"] == "ctx 0"  # input not mutated


def test_blank_and_normal():
    samples = _samples(5)
    blank = apply_context_mode(samples, "blank")
    assert all(s["context"] == "" for s in blank)
    assert [s["answer"] for s in blank] == [s["answer"] for s in samples]
    assert apply_context_mode(samples, "normal") is samples


def test_degenerate_inputs_raise():
    with pytest.raises(ValueError):
        derange_contexts(["same", "same", "same"], 0)
    with pytest.raises(ValueError):
        apply_context_mode(_samples(3), "scrambled")
