# Revision protocol: size, composition, leakage, evidence dependence, transfer

This extends the original protocol (`docs/experiment_protocol.md`) with controls that
separate *data quantity*, *data composition* and *selection leakage*, and with probes
for context use and out-of-distribution transfer.  The detector is the instrument;
the contribution is the controlled evaluation.

## 0. Facts about the shipped data (audited)

* Source rows = C1 = 10,000 rows, IDs `c1r*`. Target rows = the 4,000 rows C5 adds
  (`c2r*`-`c5r*`; 2,001 positive / 1,999 negative).  C5's file order is **not** C1
  followed by target rows, so every subset is built by ID.
* **One exact duplicate:** train row `c2r3967` equals validation row `valr3967`
  (same question, context, answer, label). It is present in C2-C5. Raw files are left
  untouched; `data.overlap_policy: remove_lower_priority` (now the base default)
  drops that single *train* copy during preprocessing and lists it in
  `data/processed/<cond>/removed_cross_split_overlap.jsonl`.  Effective sizes are
  therefore C2 10,999 ... C5 13,999, SWAP 9,999, TGT4K 3,999.  Validation/test rows are
  never removed, and validation-vs-test overlap is still a hard error.
* Mean answer length differs by origin (source ~29 words, target ~34): answer length
  is a candidate shortcut -- add a length-only baseline next to TF-IDF in the
  answer-only analysis.

## 1. Training conditions (33 training runs = 11 conditions x seeds 13, 42, 77)

| ID | Train rows | Validation used for checkpoint / T / tau | Config |
|---|---|---|---|
| C1-C5 | as in the paper (rerun, full logging) | target val (4,000) | `configs/conditions/c1..c5.yaml` |
| SWAP | 6,000 source + 4,000 target | target val | `swap.yaml` |
| TGT4K | 4,000 target | target val | `tgt4k.yaml` |
| STRICT | 8,000 source | **2,000 source holdout** | `strict.yaml` |
| S2K / S5K | 2,000 / 5,000 source (nested, stratified) | target val | `s2k.yaml`, `s5k.yaml` |
| ANSONLY | C5 data, context blank at train and eval | target val | `ansonly.yaml` |

Nesting: S2K < S5K < SWAP-source < STRICT-train < C1, from one seeded label-interleaved
ordering (seed 2026), so every prefix is exactly balanced. `python scripts/make_manifests.py`
regenerates `data/raw/<cond>/` and `data/splits/*.json` (ID lists + hashes) deterministically
and **aborts if any set overlaps the test split**.

Interpretation notes
* C1, S2K, S5K, SWAP, TGT4K still use *target* validation labels for checkpoint selection,
  temperature and threshold. **STRICT is the only condition free of target labels.**
  Comparing C1 with STRICT measures that leak; comparing SWAP with C5 and C1 separates
  composition from size.
* TGT4K trains for fewer optimizer steps (same `max_epochs`). `training_summary.json`
  records `n_train`, `best_global_step`, `epochs_run`; discuss steps vs data honestly.
* To split STRICT's gap into checkpoint-selection vs calibration leakage, score it on
  target validation (eval job `targetval`) and refit with
  `clinhallu analyze recalibrate --val <targetval preds> --test <test preds>`.

## 2. Context probes

`data.context_mode`: `normal | blank | shuffled` (cache identity includes it).
* `blank`: empty context. `shuffled`: seeded derangement of the context field across the
  split -- no example keeps a context with its own text, the context multiset is
  unchanged, answers/labels untouched. Answer-aware context selection then runs on the
  *wrong* context (shuffle first, select second).
* Eval-only probes on C1/C5 checkpoints: `clinhallu eval-only --config configs/conditions/c5.yaml
  --seed 13 --tag blank --context-mode blank --dataset data/raw/eval_data.jsonl`.
  Temperature and threshold are those frozen at validation; nothing is retuned.
* Do not claim token-level explanations; grounding scores are unvalidated.

## 3. External validity (no retraining, no threshold retuning)

`scripts/prepare_external.py` (HaluBench PubMedQA-style slice) and
`scripts/build_unseen_generator_set.py` (a generator other than MedHallu's). Both write a data
card and an audit CSV; hand-check ~100 / 50 rows and report the error rate. Labels of the
unseen set are by construction, not human-verified. Score with `eval-only --tag external|unseen`.

## 4. Saved-prediction analyses (CPU only)

* `clinhallu compare --a <preds seed13 seed42 seed77> --b <...>` -- paired grouped bootstrap
  on dF1 / dMCC / dAUROC (same resampled question/context groups for A, B and all seed pairs).
* `clinhallu analyze claim-count|low-prevalence|recalibrate` -- see `--help`.
  `claim_count` is a sentence-level proxy; confirm on a hand-checked sample.
* Prediction files carry `logit`, `group_key`, `source_tag`, `claim_count`, `answer_words`,
  `context_words`, `context_mode`, so every table is rebuildable from `artifacts/**/predictions/*.jsonl`.
* Judge calibration: convert judge outputs to the same record schema (`id, hallu_label,
  prediction, p_hallucinated_calibrated`, optional `logit`) and use `analyze recalibrate`,
  fitting only on validation answers. (Baseline-judge output conversion is not automated here.)

## 5. Checkpoints and provenance

Checkpoints store LoRA + head weights only (`checkpoints.trainable_only: true`); the frozen
DeBERTa is rebuilt from the pinned revision and verified (`encoder_revision` is stored and
checked). Legacy full checkpoints still load. Manifests and `environment.json` record
`git_commit` (set `GIT_COMMIT` on Colab, where `.git` is absent).

## 6. Pre-registered decision table (write into ANALYSIS_PLAN.md before new results)

| Result | Reading |
|---|---|
| SWAP ~ C5, source scaling flat | target composition drives the gain |
| SWAP ~ C1 | size matters; reword the "gap" claim |
| TGT4K ~ C5 | source data adds little |
| STRICT << C1 | target validation labels were helping C1 |
