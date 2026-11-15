# Revision changes (relative to the uploaded ClinHallu-main.zip)

## Repairs to the uploaded files (they did not import)
* `config/schema.py`: `class ConfigError` had no body (IndentationError) -> added docstring.
* `data/resumable_sampler.py`: mis-indented `__init__` (IndentationError) -> fixed indentation.
* `models/components/grounding.py`: `GroundingAndEvidencePooling` docstring lost its opening quotes -> restored.
* Data: train row `c2r3967` duplicates validation row `valr3967` (C2-C5). `validate_fixed_splits` now honours
  `overlap_policy`; base config default is `remove_lower_priority` (train copy dropped, recorded). Val/test overlap stays fatal.

## New
* `data/revision_splits.py`, `scripts/make_manifests.py`, `configs/conditions/{s2k,s5k,swap,tgt4k,strict,ansonly}.yaml`
  and generated `data/raw/{s2k,s5k,swap,tgt4k,strict}/`, `data/splits/` (ID manifests + hashes).
* `data/context_modes.py` (+ cache identity, dataset wiring): `normal | blank | shuffled`.
* `engine/inference.py::evaluate_on`, CLI `clinhallu eval-only` (any dataset/tag/context mode, frozen T and tau).
* Prediction files: `logit`, `group_key`, `source_tag`, `claim_count`, `answer_words`, ...; validation predictions and
  `metrics/training_summary.json` are now saved.
* `engine/checkpoints.py`: trainable-only checkpoints (`checkpoints.trainable_only`), scope-aware strict loading,
  encoder-revision check; legacy full checkpoints still load.
* `git_commit` in manifests/environment.
* `reporting/paired_bootstrap.py` (`clinhallu compare`), `reporting/revision_analyses.py` (`clinhallu analyze`).
* `colab/` worker + job list (`scripts/colab_worker.py`, `scripts/make_jobs.py`), `ClinHallu_Colab_Worker.ipynb`.
* `data/external.py` + `scripts/prepare_external.py`, `scripts/build_unseen_generator_set.py`.
* Docs: `docs/revision_protocol.md`, `ANALYSIS_PLAN.md`, `BASELINE.md`; tests in `tests/revision/`.

## Not changed
Model architecture, loss, training loop, original C1-C5/ablation configs, baselines, reporting of the original tables.
