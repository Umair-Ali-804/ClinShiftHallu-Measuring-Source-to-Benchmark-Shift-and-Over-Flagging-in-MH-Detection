<div align="center">

# ClinShiftHallu

### Measuring Source-to-Benchmark Shift and Over-Flagging in Medical Hallucination Detection

Umair Ali · Mamoona Tasadduq · Javed Ali Khan

[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.57.6-ffcc4d?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![PubMedQA](https://img.shields.io/badge/Dataset-PubMedQA-ffcc4d)](https://huggingface.co/datasets/qiaojin/PubMedQA)
[![MedHallu](https://img.shields.io/badge/Dataset-MedHallu-ffcc4d)](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu)

</div>

![ClinShiftHallu detector](artifacts/figures/architecture.png)

## Overview

Medical hallucination detectors are often trained on automatically generated data, yet it is unclear how well they transfer to benchmarks built by a different pipeline. **ClinShiftHallu** is a controlled protocol that measures this *source-to-benchmark shift*.

Given a **question**, a **context passage** and a **candidate answer**, the **ClinShiftHallu detector** estimates whether the answer contains unsupported information. It uses a frozen DeBERTa-v3-base encoder with LoRA adapters, answer-to-context cross-attention, token-level grounding and answer/evidence interaction features, followed by validation-fitted temperature scaling and a validation-selected threshold.

**The protocol.** Five nested training conditions (C1–C5) share one validation set, one frozen test set, one calibration procedure and the same seeds. C1 trains on 10,000 PubMedQA-derived (*source*) examples; C2–C5 add MedHallu-derived (*target*) examples in steps of 1,000, up to 4,000.

**What is included.**

| Component | Description |
|---|---|
| Detector | Training, validation calibration and frozen-test evaluation |
| Ablations B–F | Five independently trained cumulative variants |
| Baselines | TF-IDF + LR, semantic similarity, zero-shot NLI, SelfCheckGPT-NLI, self-consistency, four LLM judges |
| Controlled study | 33 runs: size vs. composition, strict source holdout, answer-only and context probes |
| Tooling | Automatic resume, run verification, report and table generation |

> **Naming note.** The Python package and CLI are still named `clinhallu` (`clinhallu train ...`, `python -m clinhallu`). Only the project and paper name is ClinShiftHallu.

No datasets, API keys or trained checkpoints are distributed with this repository.

## Contents

1. [Key findings](#key-findings)
2. [Installation](#installation)
3. [Data](#data)
4. [Training and evaluation](#training-and-evaluation)
5. [Ablations](#ablations)
6. [Baselines](#baselines)
7. [Full pipeline](#full-pipeline)
8. [Extended controlled study](#extended-controlled-study)
9. [Outputs](#outputs)
10. [Results](#results)
11. [Limitations](#limitations)
12. [Citation](#citation)

## Key findings

- **The gap is large.** Source-only training (C1) trails full adaptation (C5) by 16.64 F1 and 35.91 MCC; the first 1,000 target examples recover 79.7% of the F1 gain.
- **Composition, not size.** In the follow-up rerun, a size-matched mixture (6k source + 4k target) and 4k target examples alone both match C5 within seed noise.
- **Over-flagging.** The source-only detector and all four zero-shot LLM judges have recall above precision; only target adaptation reverses this. At 5% prevalence, the adapted detector's PPV falls to 50.3%.
- **Answer text carries much of the signal.** TF-IDF on the answer alone reaches 86.23 F1; blanking the context costs 2.9 F1.
- **The evidence pathway adds little.** Over answer pooling it gains +0.27 F1 and +0.39 AUROC, and the six grounding statistics lower F1 by 1.69.

## Installation

### Google Colab

Select a GPU runtime (a T4 is sufficient) and keep the project on Google Drive so checkpoints survive resets.

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/ClinShiftHallu
%pip install -r requirements/colab.txt
%pip install -e . --no-deps
```

Keep Colab's existing PyTorch/CUDA installation; restart the runtime if pip requests it.

### Local

Python 3.10–3.13 with PyTorch installed for your CPU/CUDA setup:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/check_environment.py
python -m pytest
```

Library versions can shift numbers slightly. Record yours when comparing with the paper.

## Data

Both source datasets are public and are **not** shipped: [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (source pool) and [MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) (target pool). Both are MIT-licensed; derived files must keep attribution, upstream versions and a record of the transformation steps.

Each split is a JSONL file with one object per line:

```json
{"id": "example_001", "question": "...", "context": "...", "answer": "...", "hallu_label": 0}
```

`hallu_label` is 0 for a supported answer and 1 for a hallucinated one. IDs are unique within a split, and question/context groups must not overlap between train, validation and test.

The conditions are exact cumulative supersets, each step adding 1,000 class-balanced target records:

| Condition | Source | Target | Train file | Rows |
|---|---|---|---|---|
| C1 | 10,000 | 0 | `data/raw/c1/train.jsonl` | 10,000 |
| C2 | 10,000 | 1,000 | `data/raw/c2/train.jsonl` | 11,000 |
| C3 | 10,000 | 2,000 | `data/raw/c3/train.jsonl` | 12,000 |
| C4 | 10,000 | 3,000 | `data/raw/c4/train.jsonl` | 13,000 |
| C5 | 10,000 | 4,000 | `data/raw/train.jsonl` | 14,000 |
| Validation (shared) | 0 | 4,000 | `data/raw/val.jsonl` | 4,000 |
| Test (shared, `pqa_labeled`-derived) | 0 | 1,000 | `data/raw/eval_data.jsonl` | 1,000 |

All splits are class-balanced. If your audited counts differ, edit `data.expected_rows` in `configs/base.yaml` and `configs/conditions/cN.yaml`; do not pad with duplicates. C5 reads `data/raw/train.jsonl` because the ablation configs inherit that path from `base.yaml`.

```bash
clinhallu validate --config configs/conditions/c5.yaml
```

> **Overlap note.** The released C2–C5 training files contain one row (`c2r3967`) identical to a validation row. With the default `overlap_policy: remove_lower_priority` the train copy is dropped (recorded in `removed_cross_split_overlap.jsonl`) and raw files are unchanged. This is why those conditions have one fewer training row in the follow-up rerun.

## Training and evaluation

```bash
clinhallu condition c5 --seed 13                # prepare, train, calibrate, evaluate
clinhallu condition c5 --seed 13 --smoke-test   # short check; does not touch the test split
```

The full adaptation curve over the paper's three seeds:

```bash
for seed in 13 42 77; do
  for c in c1 c2 c3 c4 c5; do clinhallu condition $c --seed $seed; done
done
```

All conditions share preprocessing and hyperparameters (AdamW, cosine decay with 10% warmup, at most five epochs with patience two, batch size 8 × 4 accumulation); only the training data change. The checkpoint is selected on validation data. Temperature `T` minimises validation cross-entropy and threshold `τ` is swept on validation (0.05–0.95) to maximise hallucinated-class F1. Test labels never influence either.

**Resume.** Every long-running command resumes automatically: after a disconnect or `Ctrl-C`, re-run the identical command. Training resumes inside the interrupted epoch and produces weights bit-identical to an uninterrupted run. Use `--no-resume` to discard saved progress.

## Ablations

Five cumulative variants on C5, each trained independently:

| Variant | Adds |
|---|---|
| B | Fused CLS/answer representation (attention pooling); evidence features zeroed |
| C | Answer-to-context cross-attention and uniform evidence pooling |
| D | Learned token-grounding weights |
| E | Six grounding/pooling statistics |
| F | Answer/evidence difference, product and cosine interactions (full detector) |

```bash
clinhallu ablate --variants B C D E F --seed 13 --continue-on-error
```

## Baselines

Nine baseline instances are configured in `configs/baselines/suite.yaml`. Five run by default; the four paid LLM judges are disabled and are run one at a time.

| Baseline | Suite name | Threshold |
|---|---|---|
| TF-IDF + logistic regression (trained on C5) | `tfidf_lr` | validation |
| Biomedical semantic similarity | `semantic_similarity` | validation |
| Zero-shot NLI (DeBERTa) | `zero_shot_nli` | validation |
| SelfCheckGPT-NLI | `selfcheckgpt` | fixed 0.5 |
| Self-consistency | `self_consistency` | fixed vote rule |
| GPT-5 Mini | `openrouter_gpt_5_mini` | fixed 0.5 |
| DeepSeek V3.2 (Chat) | `openrouter_deepseek_chat` | fixed 0.5 |
| Llama 3.3 70B Instruct | `openrouter_llama_3_3_70b_instruct` | fixed 0.5 |
| Qwen3 32B | `openrouter_qwen3_32b` | fixed 0.5 |

```bash
clinhallu baseline run --only tfidf_lr semantic_similarity zero_shot_nli
clinhallu baseline run --only openrouter_llama_3_3_70b_instruct
```

AUROC and AUPRC do not depend on the threshold; F1, MCC, precision and recall do, so compare threshold-dependent numbers with care.

### LLM judges and answer generation

These use [OpenRouter](https://openrouter.ai/). The key is read from the environment only, never from files:

```bash
read -rs -p 'OpenRouter API key: ' OPENROUTER_API_KEY; export OPENROUTER_API_KEY
```

In Colab: `os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')`.

Judges receive only the question, context and candidate answer, and return a binary label plus a hallucination probability (decision at 0.5). `deepseek-chat` is an alias that can change version; check it with `clinhallu api models --contains deepseek` and record the result. OpenRouter may route a request to different backends, so judge numbers may not reproduce exactly.

### SelfCheckGPT and self-consistency

Both share five sampled answers per evaluation question, generated once (temperature 1.0, seeds 13–17, default model `openai/gpt-4.1-mini`):

```bash
clinhallu generate --config configs/generation/shared_five.yaml --dry-run   # count requests
clinhallu generate --config configs/generation/shared_five.yaml             # 5,000 completions
clinhallu baseline run --only selfcheckgpt self_consistency                 # local scoring, no API calls
```

Output is `data/generated/shared_five/evaluation_answers.json`. Every API response is cached; malformed responses stop the run without automatic paid retries; changing the prompt, model or data requires a new output directory.

## Full pipeline

```bash
clinhallu pipeline --condition c5 --seed 13 --dry-run   # preview
clinhallu pipeline --condition c5 --seed 13             # model, ablations, generation, baselines, report
clinhallu pipeline --condition c5 --seed 13 --status    # what is left to do
clinhallu report build                                  # tables and figures
```

Generation and baselines are seed-independent and are skipped on later seeds. Use `--stages` to select stages and `--force-stage <name>` to rerun one.

## Extended controlled study

Eleven conditions × seeds 13, 42 and 77 (33 runs) separate training-set size from data composition. See `docs/revision_protocol.md` and `ANALYSIS_PLAN.md`.

| Condition | Training data | Purpose |
|---|---|---|
| C1–C5 | 10k source + 0–4k target | Adaptation curve |
| SWAP | 6k source + 4k target | Size of C1, composition of C5 |
| TGT4K | 4k target only | No source data |
| STRICT | 8k source, 2k source holdout | No target label at any stage |
| S2K, S5K | 2k / 5k source | Source-size scaling |
| ANSONLY | C5 data, context blanked | Answer-only control |

```bash
python scripts/make_manifests.py     # build the extra training sets (data/splits/)
python scripts/make_jobs.py          # write jobs/jobs.json
clinhallu run --config configs/conditions/swap.yaml --seed 13
clinhallu eval-only --config configs/conditions/c5.yaml --seed 13 \
    --tag shuffled --context-mode shuffled --dataset data/raw/eval_data.jsonl
clinhallu verify-run --all artifacts/adaptation
python scripts/rebuild_results.py --paper results/paper_table3.json
```

**Free Colab.** Use `ClinShiftHallu_Run_One_By_One.ipynb`: run setup cells 1–3 each session, then one run cell at a time (typically 2–4 runs per T4 session). Finished runs print `DONE` and are skipped; interrupted runs resume from Drive. Evaluation-probe cells wait until their training run is `DONE`.

## Outputs

| Output | Location |
|---|---|
| Checkpoints, calibration, predictions, metrics | `artifacts/adaptation/<condition>/seed_<n>/` |
| Ablations | `artifacts/ablations/<variant>/seed_<n>/` |
| Baselines | `artifacts/baselines/<instance>/` |
| Shared sampled answers | `data/generated/shared_five/` |
| Tables, figures, summaries | `artifacts/reports/` and `results/` |

Metrics are stored as fractions (paper tables show percentages). They include accuracy, hallucinated-class precision/recall/F1, MCC, AUROC and AUPRC, plus Brier score and ECE where probabilities exist.

## Results

Frozen 1,000-example `pqa_labeled`-derived test set; mean ± SD over seeds 13, 42 and 77 (%). The original runs and the follow-up rerun used different software stacks and one replaced duplicate training row, so they are reported separately and should not be pooled.

### Original runs

| Condition | F1 | MCC | AUROC |
|---|---|---|---|
| C1 (source only) | 77.11 ± 1.90 | 51.80 ± 4.79 | 83.82 ± 2.24 |
| C2 (+1k target) | 90.38 ± 0.50 | 80.88 ± 0.95 | 96.76 ± 0.04 |
| C3 (+2k target) | 92.08 ± 0.49 | 84.64 ± 1.03 | 97.75 ± 0.18 |
| C4 (+3k target) | 92.77 ± 0.25 | 85.97 ± 0.70 | 98.10 ± 0.09 |
| **C5 (+4k target)** | **93.75 ± 0.10** | **87.71 ± 0.35** | **98.49 ± 0.21** |

- Source-only training costs 16.64 F1 and 35.91 MCC; the first 1,000 target examples recover 79.7% of the F1 gain.
- Recall exceeds precision for the source-only detector and all four LLM judges; C5 reverses this (recall − precision = −2.61 points).
- At 5% hallucination prevalence, C5's PPV is 50.3% versus 11.0–27.2% for the judges.
- The strongest baseline is Llama-3.3-70B at 89.30 F1.
- Ablation: the full evidence pathway (F − B) adds +0.27 F1, +0.75 MCC and +0.39 AUROC; adding the six grounding statistics (D → E) lowers F1 by 1.69.

### Follow-up rerun (33 runs)

| System | Train rows | F1 | MCC | AUROC |
|---|---|---|---|---|
| S2K | 2,000 | 68.05 ± 1.03 | 19.77 ± 3.14 | 67.56 ± 0.52 |
| S5K | 5,000 | 72.44 ± 2.90 | 36.69 ± 9.43 | 76.05 ± 3.87 |
| STRICT | 8,000 | 70.41 ± 7.25 | 45.63 ± 8.51 | 80.94 ± 3.78 |
| C1 | 10,000 | 76.33 ± 1.35 | 50.15 ± 3.53 | 83.02 ± 1.38 |
| TGT4K (target only) | 3,999 | 93.43 ± 0.43 | 87.03 ± 0.69 | 98.21 ± 0.08 |
| SWAP | 9,999 | 93.54 ± 0.44 | 87.21 ± 0.92 | 98.43 ± 0.28 |
| C5 | 13,999 | 93.92 ± 0.58 | 88.23 ± 1.08 | 98.41 ± 0.25 |
| ANSONLY | 13,999 | 91.03 ± 0.46 | 82.21 ± 1.01 | 96.24 ± 0.41 |

- SWAP and TGT4K are statistically indistinguishable from C5, while TGT4K beats C1 by +17.1 F1 [95% CI +14.4, +19.6] with 6,000 fewer rows.
- STRICT (no target label at any stage) reaches 70.41 F1 with high seed variance; it also uses fewer training rows and a source-only validation set.
- TF-IDF on the answer text alone reaches 86.23 F1, and blanking the context costs 2.9 F1 (ANSONLY).

### Figures

| | |
|---|---|
| ![Adaptation](artifacts/figures/fig_adaptation.png) | ![Shift signature](artifacts/figures/fig_shift_signature.png) |
| Adaptation curve (C1–C5) | Over-flagging and where the gap appears |
| ![Controls](artifacts/figures/fig_controls.png) | ![Prevalence](artifacts/figures/fig_prevalence.png) |
| Size vs. composition controls | Precision at lower prevalence |

The remaining figures (`fig_baselines`, `fig_ablation`, `fig_shortcuts`) are in [`artifacts/figures/`](artifacts/figures/).

## Limitations

- Condition is confounded with training-set size in C1–C5; the follow-up controls address this partly, and no 14,000-example source-only control could be built (only 10,000 source examples exist).
- Every condition except STRICT uses 4,000 target validation labels for checkpoint selection, calibration and thresholding.
- The original checkpoints were not evaluated with blanked or shuffled context, so their dependence on the evidence is untested.
- Baselines are single runs; judges use a fixed threshold and uncalibrated probabilities, whereas the detector is temperature-scaled on target validation data.
- The hallucinated test answers come from an automated pipeline and are not clinician-verified. Results are benchmark-relative and are **not** evidence of clinical readiness.

## Citation

```bibtex
@misc{ali2026clinshifthallu,
  title  = {ClinShiftHallu: Measuring Source-to-Benchmark Shift and Over-Flagging in Medical Hallucination Detection},
  author = {Ali, Umair and Tasadduq, Mamoona and Khan, Javed Ali},
  year   = {2026}
}
```

Update the entry with the venue and DOI once the paper is published.
