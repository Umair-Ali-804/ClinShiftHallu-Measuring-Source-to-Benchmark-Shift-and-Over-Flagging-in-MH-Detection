"""Context presentation modes for the evidence-dependence probes (torch-free).

``normal``   the real context
``blank``    empty context (question + answer only)
``shuffled`` every example receives a *different* example's context

Questions, answers and labels are never modified.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

CONTEXT_MODES = ("normal", "blank", "shuffled")
DEFAULT_SHUFFLE_SEED = 20260


def context_mode_from_config(cfg: Mapping) -> Tuple[str, int]:
    """Return ``(context_mode, shuffle_seed)`` from a resolved project config."""
    data = cfg.get("data", {})
    mode = str(data.get("context_mode", "normal")).lower()
    if mode not in CONTEXT_MODES:
        raise ValueError(f"data.context_mode must be one of {CONTEXT_MODES}, got {mode!r}")
    return mode, int(data.get("context_shuffle_seed", DEFAULT_SHUFFLE_SEED))


def derange_contexts(contexts: Sequence[str], seed: int) -> List[int]:
    """Deterministic assignment ``perm`` such that example *i* receives the context
    of example ``perm[i]``.

    ``perm`` is a permutation (every context is used exactly once, so the
    context length distribution is unchanged) and no example keeps a context
    whose text equals its own -- not merely its own index, so duplicated
    contexts cannot slip through.
    """
    n = len(contexts)
    if n < 2 or len(set(contexts)) < 2:
        raise ValueError("Shuffling contexts needs at least two distinct contexts")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    perm = np.empty(n, dtype=np.int64)
    perm[order] = np.roll(order, -1)  # a single cycle: no fixed points
    for _ in range(200):
        bad = [i for i in range(n) if contexts[perm[i]] == contexts[i]]
        if not bad:
            break
        for i in bad:
            for _attempt in range(100):
                j = int(rng.integers(n))
                if (
                    j != i
                    and contexts[perm[j]] != contexts[i]
                    and contexts[perm[i]] != contexts[j]
                ):
                    perm[i], perm[j] = perm[j], perm[i]
                    break
    if any(contexts[perm[i]] == contexts[i] for i in range(n)):
        raise RuntimeError("Could not construct a context derangement; data has too many duplicates")
    return [int(v) for v in perm]


def apply_context_mode(samples: List[Dict], mode: str, seed: int = DEFAULT_SHUFFLE_SEED) -> List[Dict]:
    """Return samples with the context field replaced according to ``mode``.

    Questions, answers and labels are never touched.  ``blank`` empties the
    context; ``shuffled`` gives each example another example's context (answer
    aware context selection then runs against that wrong context).
    """
    if mode not in CONTEXT_MODES:
        raise ValueError(f"Unknown context mode {mode!r}; choose from {CONTEXT_MODES}")
    if mode == "normal":
        return samples
    out = [dict(sample) for sample in samples]
    if mode == "blank":
        for sample in out:
            sample["context"] = ""
            sample["context_mode"] = "blank"
        return out
    contexts = [str(sample["context"]) for sample in samples]
    perm = derange_contexts(contexts, seed)
    for index, sample in enumerate(out):
        donor = samples[perm[index]]
        sample["context"] = donor["context"]
        sample["context_mode"] = "shuffled"
        sample["context_donor_id"] = donor.get("sample_id", donor.get("id"))
    return out
