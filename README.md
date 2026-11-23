# ClinShiftHallu: Measuring Source-to-Benchmark Shift and Over-Flagging in Medical Hallucination Detection

[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.57.6-ffcc4d)](https://huggingface.co/docs/transformers)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM%20judges-6467f2?logo=openai&logoColor=white)](https://openrouter.ai/)

Code for the paper **"ClinShiftHallu: Measuring Source-to-Benchmark Shift and Over-Flagging in Medical Hallucination Detection"** (Umair Ali, Mamoona Tasadduq, Javed Ali Khan).
Repository: <https://github.com/Umair-Ali-804/ClinShiftHallu-Measuring-Source-to-Benchmark-Shift-and-Over-Flagging-in-MH-Detection.git>

**Datasets**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-PubMedQA-ffcc4d)](https://huggingface.co/datasets/qiaojin/PubMedQA)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MedHallu-ffcc4d)](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu)

## Contents

1. [What this project is](#1-what-this-project-is)
2. [Install](#2-install)
3. [Data: what the paper used and how to build it](#3-data-what-the-paper-used-and-how-to-build-it)
4. [Training conditions](#4-training-conditions)
5. [Run the original C1–C5 experiments](#5-run-the-original-c1c5-experiments)
6. [Run the 33-run follow-up grid](#6-run-the-33-run-follow-up-grid)
7. [Evaluation probes and analyses](#7-evaluation-probes-and-analyses)
8. [Ablations B–F](#8-ablations-bf)
9. [Baselines](#9-baselines)
10. [Shared five-answer generation and SelfCheckGPT-NLI](#10-shared-five-answer-generation-and-selfcheckgpt-nli)
11. [Reports, caches and cost control](#11-reports-caches-and-cost-control)
12. [Verify the installation](#12-verify-the-installation)
13. [Results reported in the paper](#13-results-reported-in-the-paper)
14. [Limitations and scope](#14-limitations-and-scope)

---

## 1. What this project is

ClinShiftHallu is a controlled **protocol** for measuring what a medical hallucination detector loses when its training data (automatically generated, PubMedQA-derived) and its evaluation data (MedHallu-derived) come from different construction pipelines. The paper calls this effect *source-to-benchmark shift*, not domain shift, because both pools share a clinical domain and an upstream corpus (PubMedQA) and differ in how examples were built.

Two things are kept separate:

- **The ClinShiftHallu protocol**: five nested training conditions (C1–C5), one shared validation set, one frozen test set, one calibration procedure, three seeds (13, 42, 77), plus follow-up controls (Section 4).
- **The ClinShiftHallu detector**: the model used to run the protocol. It encodes question, context and answer jointly with a frozen DeBERTa-v3-base + LoRA, pools the answer, builds answer-to-context evidence vectors, compares them with agreement/difference features, and calibrates with temperature scaling `T` and a threshold `τ` fitted on validation data. The architecture is not claimed as the main contribution. Ablation variants B–F (Section 8) evaluate its components.

Terminology:

| Term | Meaning |
|---|---|
| Source | PubMedQA-derived, automatically generated training data |
| Target | MedHallu-derived data (training increments, validation, test) |
| Source-only (C1) | No target examples for gradient updates. Target validation labels are **still** used for checkpoint selection, temperature and threshold in every condition except STRICT |
| Over-flagging | More false positives than false negatives, equivalently recall > precision (R − P > 0) |
| Gap | F1/MCC difference on the same frozen test set between C1 and C5 |

**Every long-running command resumes automatically.** If training, an ablation, a baseline or the pipeline is interrupted (Colab disconnect, closed laptop, `Ctrl-C`), re-run the identical command and it continues from where it stopped, including inside the epoch it was in. See [resume.md](docs/resume.md).

There are no research datasets, API keys or trained checkpoints in this repository. Results are reported in the paper (Section 13), not shipped as files. Running the follow-up grid produces per-example prediction files, fitted temperatures and thresholds, and the scripts rebuild every table and figure of the paper's follow-up section. See [validation_report.md](docs/validation_report.md) for the local checks and their limits.

---

## 2. Install

Run commands from the directory that contains `pyproject.toml`, `src/`, `configs/` and `data/`.

### Google Colab

Select **Runtime → Change runtime type → GPU**. A free T4 is enough (the paper's runs used a Tesla T4). Keep the project on Google Drive so checkpoints and caches survive a Colab reset.

```python
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/ClinShiftHallu'   # folder containing pyproject.toml, src/, configs/, data/
import os
assert os.path.exists(f'{ROOT}/pyproject.toml'), f'No project at {ROOT}'
```

```python
!nvidia-smi -L
!pip -q install -r {ROOT}/requirements/colab.txt
# If pip asks to restart: Runtime -> Restart session, then mount Drive and run this cell again.
```

Session settings (run in **every** new session):

```python
import os, datetime
os.environ['GIT_COMMIT'] = 'colab-' + datetime.date.today().isoformat()   # or your real commit hash
os.environ['CLINSHIFTHALLU_ROOT'] = ROOT
os.environ['PYTHONPATH'] = f'{ROOT}/src'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
```

Keep Colab's existing PyTorch/CUDA installation. The tested stack is Transformers 4.57.6, Tokenizers 0.22.2 and PEFT 0.18.1 (the stack of the follow-up rerun).

> **Environment note.** The paper's *original* runs used Python 3.13, CUDA 12.8, PyTorch 2.11, Transformers 5.16, PEFT 0.20, Accelerate 1.14 and NumPy 2.5 on a Tesla T4. The follow-up rerun used the Transformers 4.57.6 / PEFT 0.18.1 stack. Same-seed reruns are **not bit-identical** across stacks (per-seed F1 differed by up to 2.29 points), so record your versions and do not pool original and rerun numbers.

In a Colab cell put `!` before each shell command, or `%%bash` on the first line of a multi-command cell.

### Local Python

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/check_environment.py
```

`python -m clinshifthallu` can replace `clinshifthallu` in every command below.

---

## 3. Data: what the paper used and how to build it

### 3.1 Source datasets

| | Dataset | Role |
|---|---|---|
| [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-PubMedQA-ffcc4d)](https://huggingface.co/datasets/qiaojin/PubMedQA) | **PubMedQA** | Biomedical QA over PubMed abstracts (question from the title, context from the abstract *excluding* the conclusion, long answer from the conclusion, yes/no/maybe label). Upstream corpus for both pools; the 10,000-example **source** pool is PubMedQA-derived. Generation lineage for the source examples was not retained. |
| [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MedHallu-ffcc4d)](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) | **MedHallu** | Medical hallucination benchmark over PubMedQA. Each item has a supported and a hallucinated answer. Supplies the **target** pool. |

Neither dataset ships with this repository. Download them, build the splits below, and check licence terms before redistributing derived files (both upstream projects are MIT-licensed; derived files must restore attribution, upstream versions and a record of the transformation steps).

### 3.2 Which MedHallu subset is used where

| Use | MedHallu subset | Notes |
|---|---|---|
| Target training increments (4,000) | `pqa_artificial` | one existing answer kept per distinct question |
| Validation (4,000) | `pqa_artificial` | disjoint from the training pool |
| **Test (1,000)** | `pqa_labeled` | built over PubMedQA's 1,000 expert-annotated instances |

The test set is called the **"MedHallu `pqa_labeled`-derived test set"**, not an "expert test set": PubMedQA's M.D. annotators labelled the original yes/no/maybe task, but MedHallu's hallucinated answers were produced by an automated pipeline and are not reported as clinician-verified.

### 3.3 How the target splits were sampled

- 8,000 distinct questions were randomly selected from `pqa_artificial`, keeping **one** existing answer per question: supported (label 0) for 4,000 questions and hallucinated (label 1) for the other 4,000.
- These 8,000 were split into a **target training pool** and a **validation set**, each with 2,000 supported + 2,000 hallucinated.
- The training pool was divided into four class-balanced increments of 1,000 (500 supported + 500 hallucinated), added cumulatively in C2–C5.
- The increment size was fixed before training. Smaller increments were not evaluated.

### 3.4 File layout

```text
data/raw/c1/train.jsonl      10,000 rows  (source only)
data/raw/c2/train.jsonl      11,000 rows
data/raw/c3/train.jsonl      12,000 rows
data/raw/c4/train.jsonl      13,000 rows
data/raw/train.jsonl         14,000 rows  (C5)
data/raw/val.jsonl            4,000 rows  (shared)
data/raw/eval_data.jsonl      1,000 rows  (shared, frozen test)
```

C5 deliberately reads `data/raw/train.jsonl` (not a `c5/` folder) because the ablation configs inherit their training path from `configs/base.yaml`. Moving C5's data without also editing `configs/base.yaml` makes `clinshifthallu ablate` stop with *"Data registered as 'c5' changed"*. If you move data, update the model config, `configs/baselines/suite.yaml` and `configs/generation/shared_five.yaml` consistently.

Each line is a JSON object:

```json
{"id":"example_001","question":"What happened?","context":"The patient improved after treatment.","answer":"The patient improved.","hallu_label":0}
```

Required fields: `id` (string or integer), `question`, `context`, `answer` (nonempty strings) and `hallu_label` (0 = supported, 1 = hallucinated). The paper's records carry **only these five fields**: no category, difficulty or annotator metadata, so no error-type breakdown is possible (upstream MedHallu has both; they could be recovered by matching questions). Optional metadata such as `group_id`, `difficulty`, `generation_model` and `category` is preserved where supported. IDs must be unique within a split.

Records sharing a question are treated as groups and must stay in the same split. Question/context groups must not overlap across train, validation and test. Repeated questions are allowed inside a training split; validation and test contain none.

Context is answer-aware: up to eight sentences and roughly 280 context tokens are kept, scored by question similarity (0.45), answer similarity (0.45) and entity overlap (0.10).

All splits are class-balanced (positive rate 0.500; test = 500 supported / 500 hallucinated). Each split must contain both classes. Set the audited counts in `configs/base.yaml` and `configs/conditions/cN.yaml` (`data.expected_rows.train`) if your counts differ. Do not pad with duplicates. All runs save dataset hashes and reject changes to an already registered protocol.

Validate a condition before training:

```bash
clinshifthallu validate --config configs/conditions/c5.yaml
```

### 3.5 Split statistics (released files, original runs)

Word counts are whitespace-delimited, not tokens. `Dup. Q rows` counts rows beyond the first occurrence of a question (including questions shared between source and target).

| Split | Class 0 | Class 1 | Unique Q | Dup. Q rows | Context words (min/mean/med/P95/max) | Answer words (min/mean/med/P95/max) |
|---|---|---|---|---|---|---|
| C1 train | 5,000 | 5,000 | 9,997 | 3 | 2/198.78/198/286/641 | 2/29.34/26/58/168 |
| C2 train | 5,500 | 5,500 | 10,944 | 56 | 2/200.03/199/289/641 | 2/29.78/26/59/168 |
| C3 train | 6,000 | 6,000 | 11,895 | 105 | 2/201.28/200/290/641 | 2/30.07/26/60/262 |
| C4 train | 6,500 | 6,500 | 12,847 | 153 | 2/202.34/202/291/641 | 1/30.34/26/61/262 |
| C5 train | 7,000 | 7,000 | 13,797 | 203 | 2/203.21/202/293/641 | 1/30.65/26/62/262 |
| Validation | 2,000 | 2,000 | 4,000 | 0 | 15/213.30/212/303/634 | 4/33.30/28/68/253 |
| Test (`pqa_labeled`) | 500 | 500 | 1,000 | 0 | 49/212.10/212/307.05/421 | 9/34.55/29.5/69/133 |

Supported answers are longer than hallucinated ones in every split. On the test set the means are 40.2 vs 28.9 words (medians 37 vs 23), and answer word count alone gives AUROC 73.0. Every one of the 1,000 test answers is a single sentence.

### 3.6 Leakage audit and the one known duplicate

- Exact ID, question, context, answer and (q, c, a) tuple overlap between the 4,000 target training records and the test set: **0** for every field.
- Character 3–5-gram TF-IDF nearest-neighbour screen (target training vs test): max cosine 0.7392 (question), 0.6704 (answer), 0.7036 (question + answer), 0.7822 (context + answer). **No pair reaches 0.80.** Manual inspection found related topics or generic study phrasing, not duplicates.
- Question-level overlap in the released files:

| Pair | Rows sharing a question | Distinct questions | Shared question + context |
|---|---|---|---|
| Source vs target training | 200 | 200 | 0 |
| Source vs validation | 223 | 222 | 0 |
| Source vs test | 0 | 0 | 0 |
| Target training vs validation | 1 | 1 | 1 |
| Target training vs test | 0 | 0 | 0 |
| Validation vs test | 0 | 0 | 0 |

- **Known duplicate.** One target training record (`c2r3967`, first target increment, therefore present in C2–C5 as released) is identical to one validation record. The **original runs used it**. The follow-up reruns replace it with another unique target example so planned sizes stay unchanged (C2 = 11,000 … C5 = 14,000). In this repository the default `overlap_policy: remove_lower_priority` drops that single train copy (listed in `removed_cross_split_overlap.jsonl`); raw files are unchanged.
- Source and validation share 222 questions (each with a different context), so validation is not fully independent of the source data for C1 and the other source-only conditions.
- These checks do not detect semantic or derivational overlap, and without upstream lineage records they cannot establish the independence of the original source records.

---

## 4. Training conditions

### 4.1 Original protocol: C1–C5 (exact cumulative supersets)

Each step keeps every earlier record and adds exactly 1,000 balanced target records.

| Condition | Source | Target | Total train | Target labels consumed (train + 4,000 val) |
|---|---|---|---|---|
| C1 | 10,000 | 0 | 10,000 | 4,000 |
| C2 | 10,000 | 1,000 | 11,000 | 5,000 |
| C3 | 10,000 | 2,000 | 12,000 | 6,000 |
| C4 | 10,000 | 3,000 | 13,000 | 7,000 |
| C5 | 10,000 | 4,000 | 14,000 | 8,000 |
| Validation (shared) | 0 | 4,000 | 4,000 | n/a |
| Test (shared) | 0 | 1,000 | 1,000 | n/a |

Because validation is drawn from the target distribution in **every** condition (including C1), each condition gets its own best temperature and threshold by the same procedure. Differences between conditions therefore reflect the training mixture (and the accompanying growth in training-set size), not the calibration step.

### 4.2 Follow-up controls (33 runs = 11 conditions × 3 seeds)

These separate training-set size from data composition, test a pipeline with no target labels, and test how much signal sits in the answer text. The source subsets are nested (S2K ⊂ S5K ⊂ SWAP-source ⊂ STRICT-train ⊂ C1) and drawn from one seeded, label-interleaved ordering, so every subset is exactly class-balanced.

| Condition | Source | Target | Total | Validation used | Purpose |
|---|---|---|---|---|---|
| C1 | 10,000 | 0 | 10,000 | target, 4,000 | reference (source only) |
| C5 | 10,000 | 4,000 | 14,000 | target, 4,000 | reference (full adaptation) |
| SWAP | 6,000 | 4,000 | 10,000 | target, 4,000 | size of C1, composition of C5 |
| TGT4K | 0 | 4,000 | 4,000 | target, 4,000 | target data only |
| STRICT | 8,000 | 0 | 8,000 | **source holdout, 2,000** | no target label at any stage |
| S2K | 2,000 | 0 | 2,000 | target, 4,000 | source scaling |
| S5K | 5,000 | 0 | 5,000 | target, 4,000 | source scaling |
| ANSONLY | 10,000 | 4,000 | 14,000 | target, 4,000 | context blanked in training **and** test |

A 14,000-example source-only control cannot be built because only 10,000 source examples exist, so SWAP matches C1's size instead. STRICT is the only condition that uses no target label anywhere before the frozen test evaluation.

### 4.3 Hyperparameters (identical across all conditions and seeds)

| Item | Value |
|---|---|
| Encoder (frozen) | `microsoft/deberta-v3-base`, hidden size 768, sequence length 512 |
| Question / answer budgets | 96 / 128 tokens (context ≈ 280 tokens, 8 sentences) |
| LoRA | rank 16, α 32, dropout 0.05, on Q, K, V and attention output |
| Learning rates | LoRA 2e-5, new modules 1e-4 |
| Optimiser / schedule | AdamW, weight decay 0.01, cosine decay, 10% warmup |
| Gradient clipping | 1.0 |
| Batch size | 8 × 4 accumulation (effective 32) |
| Epochs / patience | at most 5 / 2 |
| Attention-pooling hidden size / cross-attention heads | 256 / 8 |
| Classifier | 3079 → 1024 → 512 → 128 → 1, GELU, LayerNorm after the 1024 and 512 layers, dropout 0.20 / 0.15 / 0.10 |
| Grounding temperature `T_g` | 1.0 |
| Checkpoint selection | validation F1 (hallucinated class) at the F1-optimal threshold, each epoch |
| Calibration | `T` minimises validation cross-entropy; `τ` swept 0.05–0.95 (step 0.01, ties to higher recall) to maximise hallucinated-class F1. Test labels never influence either |
| ECE | 15 equal-width bins on the hallucinated-class probability |
| Precision | mixed precision and gradient checkpointing enabled |
| Seeds | 13, 42, 77 |

Parameter budget: about 11.0M nominally trainable (11,035,650). An unused reverse context-to-answer branch (about 2.36M) receives no gradient, so about **8.67M** are actually updated. The DeBERTa backbone is frozen.

---

## 5. Run the original C1–C5 experiments

### 5.1 One condition, one seed

Prepare, train, calibrate on validation, and evaluate the frozen test split, with automatic resume:

```bash
clinshifthallu condition c5 --seed 13
```

Other conditions, once their training files are in place and their audited row counts are set:

```bash
clinshifthallu condition c1 --seed 13
clinshifthallu condition c2 --seed 13
clinshifthallu condition c3 --seed 13
clinshifthallu condition c4 --seed 13
```

The full C1–C5 curve over the paper's three seeds:

```bash
for seed in 13 42 77; do
  for c in c1 c2 c3 c4 c5; do
    clinshifthallu condition $c --seed $seed
  done
done
```

Smoke check (short training run, does not evaluate the frozen test split):

```bash
clinshifthallu condition c5 --seed 13 --smoke-test
```

### 5.2 Step by step

```bash
clinshifthallu prepare  --config configs/conditions/c5.yaml --seed 13
clinshifthallu train    --config configs/conditions/c5.yaml --seed 13
clinshifthallu train    --config configs/conditions/c5.yaml --seed 13 --resume
clinshifthallu evaluate --config configs/conditions/c5.yaml --seed 13
clinshifthallu evaluate --config configs/conditions/c5.yaml --seed 13 --force   # recompute; nothing is selected on test labels
```

To discard saved progress and start from scratch:

```bash
clinshifthallu condition c5 --seed 13 --no-resume
```

Resume restores model, optimizer, scheduler, scaler, epoch, early-stopping state, the position inside the epoch, partial epoch metrics, and the RNG state of Python, NumPy, PyTorch CPU and CUDA. Training order for epoch *k* is derived from `(seed, k)`, so a resumed run produces bit-identical weights to an uninterrupted run on the same software stack (enforced by a test).

### 5.3 The whole pipeline

Preview (no training, data requests or paid completions):

```bash
clinshifthallu pipeline --condition c5 --seed 13 --dry-run
```

Run everything, per seed (each seed gets its own directory):

```bash
for seed in 13 42 77; do
  clinshifthallu pipeline --condition c5 --seed $seed
done
```

Stages run sequentially: (1) prepare, train, calibrate and evaluate the detector; (2) ablations B–F; (3) generate five sampled answers per evaluation question; (4) run the five default-enabled baselines (three conventional, SelfCheckGPT-NLI, self-consistency; the four paid judges are run explicitly); (5) build tables, summaries and figures. Generation and baselines are seed-independent, so later seeds skip them as completed stages.

```bash
clinshifthallu status
clinshifthallu pipeline --condition c5 --seed 13 --status
clinshifthallu pipeline --condition c5 --seed 13 --force-stage baselines
clinshifthallu pipeline --condition c5 --seed 13 --no-resume          # every stage again
clinshifthallu pipeline --condition c5 --seed 13 --stages model report
clinshifthallu pipeline --stages generate baselines report
```

The supplied ablation configs use C5. For C1–C4, omit the `ablations` stage or use the individual condition commands above.

---

## 6. Run the 33-run follow-up grid

### 6.1 First session only: build the extra training sets and job list

```bash
python scripts/make_manifests.py     # S2K, S5K, SWAP, TGT4K, STRICT + ID manifests (data/splits/)
python scripts/make_jobs.py          # jobs/jobs.json + jobs_A1..A5.txt
```

In Colab:

```python
if not os.path.exists(f'{ROOT}/jobs/jobs.json'):
    !cd {ROOT} && python scripts/make_manifests.py > /dev/null && python scripts/make_jobs.py
```

Optional first-session checks (if either fails, fix it before starting the runs):

```bash
python -m pytest tests -q -x -p no:cacheprovider
clinshifthallu run --config configs/conditions/c1.yaml --seed 13 --smoke-test
```

### 6.2 A single condition

```bash
clinshifthallu run --config configs/conditions/swap.yaml --seed 13
```

### 6.3 One run per Colab cell (free ~2-hour sessions)

Every new Colab session: run the mount, install and session-settings cells (Section 2), then run the next unfinished run cell. A finished run writes a `DONE` marker and re-running its cell just skips it. If a session ends mid-run, run the **same cell** again: it resumes from the last checkpoint copied to Drive (every 10 minutes). A free T4 session usually fits 2–4 runs. Suggested order: all 11 conditions for seed 13, then seed 42, then seed 77.

```bash
python scripts/colab_worker.py --drive-root /content/drive/MyDrive/ClinShiftHallu --account SOLO --run c1_seed13 --max-hours 1.9
```

Run ids are `<condition>_seed<N>` with conditions `c1 c2 c3 c4 c5 swap tgt4k strict s2k s5k ansonly` and seeds `13 42 77`:

```bash
for seed in 13 42 77; do
  for run in c1 c2 c3 c4 c5 swap tgt4k strict s2k s5k ansonly; do
    python scripts/colab_worker.py --drive-root /content/drive/MyDrive/ClinShiftHallu \
        --account SOLO --run ${run}_seed${seed} --max-hours 1.9
  done
done
```

Rough single-run times on a T4 from the notebook (placeholders; replace with your own): C1 26 min, C2 29, C3 31, C4 34, C5 36, SWAP 26, TGT4K 10, STRICT 21, S2K 8, S5K 13, ANSONLY 36.

With up to five free Colab accounts, use `--account A1` … `A5` and the matching `jobs_A1.txt` … `jobs_A5.txt`; 
### 6.4 Status: which of the 33 are finished?

```python
import json, os
jobs = json.load(open(f'{ROOT}/jobs/jobs.json'))
def done(j):
    d = f"{ROOT}/artifacts/adaptation/{j['condition']}/seed_{j['seed']}"
    return os.path.exists(f"{d}/DONE" if j['kind']=='train' else f"{d}/eval_{j['tag']}.DONE")
train = [j for j in jobs if j['kind']=='train']
print(f"training {sum(map(done, train))}/33 done")
print('FINISHED :', [j['run_id'] for j in train if done(j)])
print('NEXT     :', next((j['run_id'] for j in train if not done(j)), 'all done'))
```

---

## 7. Evaluation probes and analyses

These probes test the **frozen checkpoints** with blank or shuffled context, and score STRICT on the target validation split. The paper lists them as future work (it reports only the retrained ANSONLY variant), so any numbers you get here are new. A probe waits (prints `waiting-for`) until its training run is `DONE`.

```bash
for seed in 13 42 77; do
  python scripts/colab_worker.py --drive-root /content/drive/MyDrive/ClinShiftHallu --account SOLO \
      --run eval_c1_seed${seed}_blank eval_c1_seed${seed}_shuffled --max-hours 1.9
  python scripts/colab_worker.py --drive-root /content/drive/MyDrive/ClinShiftHallu --account SOLO \
      --run eval_c5_seed${seed}_blank eval_c5_seed${seed}_shuffled --max-hours 1.9
  python scripts/colab_worker.py --drive-root /content/drive/MyDrive/ClinShiftHallu --account SOLO \
      --run eval_strict_seed${seed}_targetval --max-hours 1.9
done
```

Direct evaluation of an existing checkpoint with a changed context:

```bash
clinshifthallu eval-only --config configs/conditions/c5.yaml --seed 13 \
    --tag shuffled --context-mode shuffled --dataset data/raw/eval_data.jsonl
```

Paired grouped-bootstrap comparison of two systems (2,000 replicates in the paper; the same resample is applied to both systems and all three seeds):

```bash
clinshifthallu compare --a <swap preds...> --b <c5 preds...> --metric f1 mcc auroc
```

Claim-count analysis of per-example predictions:

```bash
clinshifthallu analyze claim-count --pred artifacts/adaptation/c5/seed_13/predictions/evaluation_predictions.jsonl
```

The paper could not test its claim-count hypothesis on this test set because every test answer is a single sentence; testing it needs multi-sentence answers.

### Check runs and rebuild the paper tables

Put the paper's Table 3 numbers (percent) into `results/paper_table3.json` first (in the format `scripts/rebuild_results.py` expects), then:

```bash
clinshifthallu verify-run --all artifacts/adaptation
python scripts/rebuild_results.py --paper results/paper_table3.json
```

List the saved checkpoints and their sizes:

```python
import glob, os
for p in sorted(glob.glob(f'{ROOT}/artifacts/adaptation/*/seed_*/checkpoints/best_model.pt')):
    print(f"{p.split('adaptation/')[1]:45s} {os.path.getsize(p)/1e6:7.0f} MB")
```

Commit [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) before any new result exists. Each follow-up run saves per-example predictions with raw scores, the fitted temperature and threshold, and the code commit (set `GIT_COMMIT`); the verify step checks that every decision equals the validation-fitted threshold applied to the calibrated probability.

---

## 8. Ablations B–F

The ablation is cumulative on C5; each rung adds one block. Variant F is the complete ClinShiftHallu detector and is the same set of runs as C5.

| Variant | Features used by its classifier |
|---|---|
| B | Fused CLS/answer representation (attention pooling); evidence features zeroed |
| C | B plus answer-to-context cross-attention and uniform evidence pooling |
| D | C plus learned token-grounding weights for evidence pooling |
| E | D plus six grounding/pooling statistics |
| F | E plus answer/evidence difference, product and cosine interactions (full detector) |

All variants jointly encode question, context and answer. B disables the explicit evidence branch; it does **not** remove context from the encoder input.

```bash
clinshifthallu ablate --variants B C D E F --seed 13      # safe to interrupt and re-run
clinshifthallu ablate --variants B C D E F --seed 13 --continue-on-error
clinshifthallu ablate --variants B --seed 13              # or one variant at a time
for seed in 13 42 77; do clinshifthallu ablate --variants B C D E F --seed $seed; done
```

Re-running skips finished variants, resumes the one in flight from its own checkpoint, and continues with the rest. `--continue-on-error` keeps going past a variant that fails outright (for example out of memory); failures are reported at the end and retried on the next run.

Paper findings (three seeds, C5):

| Variant | F1 | MCC | AUROC | AUPRC |
|---|---|---|---|---|
| B Answer pooling | 93.48 ± 0.14 | 86.96 ± 0.22 | 98.10 ± 0.11 | 98.30 ± 0.09 |
| C + Cross-attention | 94.06 ± 0.16 | 88.12 ± 0.25 | 98.20 ± 0.10 | 98.44 ± 0.10 |
| D + Token grounding | **94.64 ± 0.13** | **89.31 ± 0.21** | 98.25 ± 0.09 | 98.48 ± 0.08 |
| E + Grounding statistics | 92.95 ± 0.18 | 86.18 ± 0.27 | 98.11 ± 0.12 | 98.34 ± 0.09 |
| F + Contradiction features (full) | 93.75 ± 0.10 | 87.71 ± 0.35 | **98.49 ± 0.21** | **98.62 ± 0.15** |
| F − B | +0.27 | +0.75 | +0.39 | +0.32 |

Adding the six grounding statistics (D→E) lowers F1 by 1.69 and MCC by 3.13 and is the only step that lowers both. The whole evidence pathway (F − B) is small and not robust across three seeds (uncorrected Welch p ≈ 0.04–0.07). The released F configuration includes the statistics block because it is the configuration evaluated throughout; whether removing it from the complete model helps was not tested, and choosing D after seeing its test F1 would be selecting on the test set.

---

## 9. Baselines

`configs/baselines/suite.yaml` contains nine configured entries. Five local/scoring entries are enabled by default; the four paid judges are disabled by default to prevent accidental bulk spending. `--only` runs a selected entry even if it is disabled by default.

```bash
clinshifthallu baseline list
clinshifthallu baseline run --dry-run        # prints the planned commands only
clinshifthallu baseline run                  # all enabled baselines
clinshifthallu baseline summarize            # rebuild baseline reports only
```

### 9.1 Conventional baselines (local, no OpenRouter)

```bash
clinshifthallu baseline run --only tfidf_lr             # TF-IDF + logistic regression, trained on the 14,000-row C5 split
clinshifthallu baseline run --only semantic_similarity  # PubMed sentence encoder, weakest-supported claim vs context windows
clinshifthallu baseline run --only zero_shot_nli        # DeBERTa cross-encoder, contradiction weight 0 (entailment-deficit detector)
```

The semantic and NLI models may download Hugging Face weights on first use. Their thresholds are selected on validation only.

TF-IDF settings: word unigrams and bigrams (min document frequency 2, max 0.98, sublinear TF, at most 100,000 features), logistic regression with balanced class weights, threshold that maximises F1 on validation. The fit is deterministic (no seed variance). The paper also refits it on single segments (question only, context only, question + answer, answer only) for the answer-only analysis (Section 13.4).

### 9.2 OpenRouter LLM judges

The code reads `OPENROUTER_API_KEY` from the process environment (it does not load `.env`). Never paste a real key into a YAML or prompt file.

```python
import os
from google.colab import userdata
os.environ['OPENROUTER_API_KEY'] = userdata.get('OPENROUTER_API_KEY')
```

```python
import getpass, os
os.environ['OPENROUTER_API_KEY'] = getpass.getpass('OpenRouter API key: ')
```

```bash
read -rs -p 'OpenRouter API key: ' OPENROUTER_API_KEY
export OPENROUTER_API_KEY
```

| Judge | Suite name | OpenRouter model ID | Notes |
|---|---|---|---|
| GPT-5 Mini | `openrouter_gpt_5_mini` | `openai/gpt-5-mini` | Minimal reasoning, 4096-token completion budget, default temperature |
| DeepSeek V3.2 (Chat) | `openrouter_deepseek_chat` | `deepseek/deepseek-chat` | Paper reports DeepSeek-V3.2. The alias can move between versions; check what it resolves to and record it |
| Llama 3.3 70B Instruct | `openrouter_llama_3_3_70b_instruct` | `meta-llama/llama-3.3-70b-instruct` | |
| Qwen3 32B | `openrouter_qwen3_32b` | `qwen/qwen3-32b` | Thinking mode disabled for the judge task |

All four judges receive an identical system prompt (annotation definition, restricted to the supplied context) and must return a JSON object with a binary `hallu_label` and `probability_hallucinated`, under a schema constraint with greedy decoding (except GPT-5 Mini). The fixed decision rule is probability ≥ 0.5. Probabilities are verbalised and left uncalibrated, so they contain ties and are not comparable in calibration to the detector's temperature-scaled probabilities. Gold labels, training examples and validation examples are never in the prompt.

```bash
clinshifthallu api models --contains gpt-5-mini           # inspect catalog, no paid completion
clinshifthallu api models --contains deepseek-chat

clinshifthallu baseline run --only openrouter_gpt_5_mini
clinshifthallu baseline run --only openrouter_deepseek_chat
clinshifthallu baseline run --only openrouter_llama_3_3_70b_instruct
clinshifthallu baseline run --only openrouter_qwen3_32b
```

GPT-5 Mini writes to `artifacts/baselines/openrouter_gpt_5_mini_v3` and intentionally does not reuse the older `v2` cache, whose 512-token requests can end with `finish_reason: length` before the JSON is emitted. Configs live under `configs/baselines/llm_judges/<suite name>.yaml`; each judge has its own output and API-cache directory.

A judge chosen directly, with a separate output directory (add `--dry-run` to see the remaining request count without calling the API):

```bash
python -m clinshifthallu.baselines.llm_judge \
  --provider openrouter \
  --model-name openai/gpt-5-mini \
  --api-config configs/api/judge_gpt5.yaml \
  --train data/raw/train.jsonl --val data/raw/val.jsonl --eval data/raw/eval_data.jsonl \
  --output-dir artifacts/baselines/my_judge
```

Evaluating existing externally produced binary judgments (matching evaluation IDs and binary `hallu_label`; binary-only imports have no calibration metrics):

```bash
python -m clinshifthallu.baselines.llm_judge \
  --provider cached_predictions \
  --model-name my_external_judge \
  --predictions path/to/judge_predictions.json \
  --train data/raw/train.jsonl --val data/raw/val.jsonl --eval data/raw/eval_data.jsonl \
  --output-dir artifacts/baselines/my_external_judge
```

Model availability can change and OpenRouter may route a request to different serving backends within one run (the paper saw eight providers for Llama-3.3-70B and nine for DeepSeek-V3.2), so judge numbers describe a routed service and may not reproduce exactly.

---

## 10. Shared five-answer generation and SelfCheckGPT-NLI

SelfCheckGPT-NLI and self-consistency share the **same** five sampled answers per unique question/context pair, generated once through OpenRouter. Neither scoring command calls OpenRouter.

Prompts: `prompts/shared_five_system.txt` and `prompts/shared_five_user.txt`, selected by `configs/generation/shared_five.yaml` (default generator `openai/gpt-4.1-mini`; change `model_name` to use another). The loader replaces `{{INPUT_JSON}}` with a JSON object containing only `question` and `context`; the candidate answer and gold label are excluded.

```bash
clinshifthallu generate --config configs/generation/shared_five.yaml --dry-run                  # count remaining requests
clinshifthallu generate --config configs/generation/shared_five.yaml                            # evaluation answers
clinshifthallu generate --config configs/generation/shared_five.yaml --max-new-requests 100     # cap spend; exit code 2 = incomplete
```

Five independent completions per unique pair at temperature 1.0 with seeds 13–17; a fresh cache for 1,000 evaluation pairs needs 5,000 successful completions. Output: `data/generated/shared_five/evaluation_answers.json`, a JSON **array** with `id`, `question` and `answer_01` … `answer_05` per row, plus a separate `generation_manifest.json` (model, prompt, request settings, input/output hashes). An incomplete split is never exported as a complete array; re-run the same command to finish.

Optional validation answers (only if you tune the SelfCheckGPT threshold on validation):

```bash
clinshifthallu generate --config configs/generation/shared_five.yaml --splits validation
clinshifthallu generate --config configs/generation/shared_five.yaml --splits validation test
```

Scoring:

```bash
clinshifthallu baseline run --only selfcheckgpt
clinshifthallu baseline run --only self_consistency
clinshifthallu baseline run --only selfcheckgpt self_consistency
```

- **SelfCheckGPT-NLI** uses the official package's `SelfCheckNLI`, comparing each candidate-answer sentence with all five samples and averaging into a document score. Fixed threshold 0.5 (as in the paper). It is a context-conditioned, proxy-generator adaptation: it measures consistency with the proxy generator, not the original one.
- **Self-consistency** (code only, **not** in the paper's baseline table): each sampled answer gets one support vote if it entails every candidate sentence with probability ≥ 0.5; score is `1 - support_votes / 5`; at least three non-support votes predict hallucination.

Optional validation-selected SelfCheckGPT threshold, after generating validation answers (also set `threshold_mode: validation` and a new `output_dir` in `configs/baselines/selfcheckgpt.yaml` for the suite to use it; never choose between thresholds using test results):

```bash
python -m clinshifthallu.baselines.selfcheckgpt \
  --train data/raw/train.jsonl --val data/raw/val.jsonl --eval data/raw/eval_data.jsonl \
  --answers-dir data/generated/shared_five \
  --threshold-mode validation \
  --output-dir artifacts/baselines/selfcheckgpt_validation_tuned
```

---

## 11. Reports, caches and cost control

```bash
clinshifthallu report build
```

| Output | Location |
|---|---|
| Condition checkpoints, calibration, predictions, metrics | `artifacts/adaptation/<condition>/seed_<N>/` |
| Independently trained B–F results | `artifacts/ablations/<variant>/seed_<N>/` |
| Baseline aligned predictions and metrics | `artifacts/baselines/<instance>/` |
| Shared five-answer JSON files and manifest | `data/generated/shared_five/` |
| Raw generation API response cache | `data/cache/openrouter/shared_five/` |
| Raw judge API response cache | `artifacts/baselines/<judge>/api_cache/` |
| Shared local NLI cache | `data/cache/baselines/shared_nli/` |
| Publication tables / figures / summaries | `artifacts/reports/tables/`, `figures/`, `summaries/` |

**Metrics** are stored as fractions (the paper shows percentages; MCC on a [−100, 100] scale). They include accuracy, hallucinated-class precision/recall/F1, macro-F1, MCC, AUROC, AUPRC (average precision), Brier and ECE (raw and temperature-scaled) where defined. Normalised consistency/similarity scores and label-only imports omit probability-calibration metrics. Baseline F1 gets 95% percentile-bootstrap intervals (10,000 replicates, grouped by question; on this test set that coincides with ordinary resampling), and exact McNemar tests are used where per-example predictions were retained. The detector's spread is a three-seed standard deviation, which is not interchangeable with a bootstrap interval.

**Threshold protocols differ by method.** The detector, TF-IDF, semantic similarity and zero-shot NLI use validation-selected thresholds; SelfCheckGPT-NLI and the four judges decide at a fixed 0.5. AUROC and AUPRC do not depend on that choice (temperature scaling is strictly increasing, so ranking is invariant; Proposition 1 in the paper).

**Caches and cost control**

- Keep the project, answer directory, raw API caches and NLI caches together. Requests are cached by their full request body.
- Raw completions are written before parsing. Malformed JSON, a refusal, truncation, missing or duplicate fields, or an inconsistent judge label/probability stops the run, with no automatic paid regeneration and no fallback label.
- Retry count defaults to **zero**. `request.rate_limit_retries` allows bounded retries on HTTP 429 only. Transport errors and timeouts may already have been charged and are not retried.
- Re-running with a cached malformed response fails at that same response without another call; remove that cache entry to retry deliberately. Changed prompt, model or token settings need a new output directory.
- Changes to data, prompts, generation model or scoring protocol require a new experiment/output directory. Never hand-edit a generated answer file while keeping its old manifest.
- Run one writer per output/cache directory.
- No training or validation data are sent to OpenRouter by default; judges only see question, context and candidate answer from the evaluation split.

---

## 12. Verify the installation

```bash
python -m pytest
python scripts/demo_resume.py      # real interrupt-and-restart of a tiny training run, no GPU or network needed
```

The tests cover training/calibration/evaluation with a tiny local DeBERTa, every B–F ablation, local NLI and sentence-embedding runtime checks, shared-answer cache tests and mocked OpenRouter transport. These are implementation checks, not medical benchmark results.

---

## 13. Results reported in the paper

### 13.1 Original runs: C1–C5

Frozen 1,000-example MedHallu `pqa_labeled`-derived test set, mean ± SD over seeds 13/42/77 (%).

| Condition | Accuracy | Precision | Recall | F1 | MCC | AUROC | AUPRC |
|---|---|---|---|---|---|---|---|
| C1 (source only) | 75.70 ± 2.52 | 73.02 ± 3.28 | 81.73 ± 0.64 | 77.11 ± 1.90 | 51.80 ± 4.79 | 83.82 ± 2.24 | 83.57 ± 2.18 |
| C2 (+1k target) | 90.43 ± 0.47 | 90.86 ± 1.03 | 89.93 ± 1.29 | 90.38 ± 0.50 | 80.88 ± 0.95 | 96.76 ± 0.04 | 96.98 ± 0.10 |
| C3 (+2k target) | 92.27 ± 0.50 | 94.36 ± 1.37 | 89.93 ± 1.15 | 92.08 ± 0.49 | 84.64 ± 1.03 | 97.75 ± 0.18 | 97.94 ± 0.18 |
| C4 (+3k target) | 92.93 ± 0.31 | 95.00 ± 1.40 | 90.67 ± 0.92 | 92.77 ± 0.25 | 85.97 ± 0.70 | 98.10 ± 0.09 | 98.28 ± 0.16 |
| C5 (+4k target) | 93.83 ± 0.15 | 95.08 ± 1.01 | 92.47 ± 0.76 | 93.75 ± 0.10 | 87.71 ± 0.35 | 98.49 ± 0.21 | 98.62 ± 0.15 |

Calibration (mean ± SD): Brier 0.1696 (C1) → 0.0485 (C5); raw ECE 0.1287 → 0.0425; calibrated ECE 0.0655 → 0.0294.

Key findings from the original runs:

- Source-only training costs **16.64 F1** and **35.91 MCC** (C1 vs C5). C1 is also less stable across seeds (F1 SD 1.90 vs 0.10).
- The first 1,000 target examples recover 79.7% of the F1 gain (13.27 F1 and 29.08 MCC points). Later increments add 1.70, 0.69 and 0.98 F1.
- Ranking survives better than decision quality: C1 still reaches 83.82 AUROC, but that is below the 87.95–93.74 range of the zero-shot judges.
- C1 over-flags (recall 81.73 vs precision 73.02; FPR 30.3%, FNR 18.3%). C5 reverses this (FPR 4.8%, FNR 7.5%).
- At 5% assumed prevalence, projected PPV is **50.3%** for C5 and 11.0%–27.2% for the four judges (holding sensitivity and specificity fixed).

### 13.2 Baselines (single runs; detector rows are three-seed means)

| Method | F1 | MCC | AUROC | AUPRC | R − P |
|---|---|---|---|---|---|
| All-positive reference | 66.67 | 0.00 | 50.00 | 50.00 | +50.00 |
| TF-IDF + LR (C5 data) | 71.06 | 32.44 | 72.54 | 70.17 | n/r |
| Semantic similarity | 63.01 | 16.98 | 60.80 | 58.10 | n/r |
| Zero-shot NLI | 59.67 | 16.25 | 62.50 | 62.30 | n/r |
| SelfCheckGPT-NLI | 74.68 | 49.00 | 83.14 | 85.36 | +1.04 |
| DeepSeek-V3.2 | 80.41 | 57.60 | 87.95 | 85.67 | +24.37 |
| GPT-5-mini | 85.69 | 69.95 | 92.99 | 91.83 | +17.29 |
| Qwen3-32B | 88.25 | 75.59 | 93.74 | 92.65 | +11.91 |
| Llama-3.3-70B | 89.30 | 78.26 | 93.01 | 91.51 | +3.33 |
| ClinShiftHallu detector C1 | 77.11 | 51.80 | 83.82 | 83.57 | +8.71 |
| ClinShiftHallu detector C5 | 93.75 | 87.71 | 98.49 | 98.62 | −2.61 |

Every zero-shot system and the source-only detector over-flag (R − P > 0); C5 is the only system with precision above recall. The strongest baseline is Llama-3.3-70B at 89.30 F1. Comparisons between a target-supervised detector and zero-shot judges are descriptive.

### 13.3 Follow-up rerun (33 runs, mean ± SD over seeds)

These come from a rerun on a different software stack and are **not** interchangeable with the original numbers above.

| Condition | Train | F1 | MCC | AUROC | Prec. | Rec. | R − P |
|---|---|---|---|---|---|---|---|
| S2K | 2,000 | 68.05 ± 1.03 | 19.77 ± 3.14 | 67.56 ± 0.52 | 54.31 | 91.13 | +36.83 |
| S5K | 5,000 | 72.44 ± 2.90 | 36.69 ± 9.43 | 76.05 ± 3.87 | 62.53 | 86.47 | +23.93 |
| STRICT | 8,000 | 70.41 ± 7.25 | 45.63 ± 8.51 | 80.94 ± 3.78 | 74.94 | 68.13 | −6.81 |
| C1 | 10,000 | 76.33 ± 1.35 | 50.15 ± 3.53 | 83.02 ± 1.38 | 72.22 | 81.00 | +8.78 |
| SWAP | 10,000 | 93.54 ± 0.44 | 87.21 ± 0.92 | 98.43 ± 0.28 | 94.37 | 92.73 | −1.64 |
| C2 | 11,000 | 90.33 ± 0.60 | 81.10 ± 1.08 | 96.59 ± 0.71 | 92.04 | 88.73 | −3.31 |
| C3 | 12,000 | 91.01 ± 1.01 | 82.68 ± 1.75 | 97.38 ± 0.39 | 93.72 | 88.47 | −5.26 |
| C4 | 13,000 | 92.75 ± 0.47 | 85.87 ± 0.92 | 98.11 ± 0.40 | 94.72 | 90.87 | −3.85 |
| C5 | 14,000 | 93.92 ± 0.58 | 88.23 ± 1.08 | 98.41 ± 0.25 | 96.23 | 91.73 | −4.49 |
| TGT4K | 4,000 | 93.43 ± 0.43 | 87.03 ± 0.69 | 98.21 ± 0.08 | 94.37 | 92.53 | −1.83 |
| ANSONLY | 14,000 | 91.03 ± 0.46 | 82.21 ± 1.01 | 96.24 ± 0.41 | 91.80 | 90.27 | −1.54 |

Paired differences (rerun, 95% paired grouped-bootstrap intervals; uncorrected for multiple comparisons):

| A − B | ΔF1 | ΔMCC | ΔAUROC |
|---|---|---|---|
| SWAP − C1 (same size, 4,000 target) | +17.2 [+14.7, +19.7] | +37.1 [+32.2, +41.9] | +15.4 [+13.1, +17.7] |
| C5 − SWAP | +0.4 [−0.3, +1.1] | +1.0 [−0.4, +2.4] | −0.0 [−0.2, +0.2] |
| C5 − TGT4K | +0.5 [−0.6, +1.5] | +1.2 [−0.9, +3.2] | +0.2 [−0.2, +0.6] |
| TGT4K − C1 | +17.1 [+14.4, +19.6] | +36.9 [+31.7, +41.8] | +15.2 [+12.8, +17.6] |
| C2 − C1 (first 1,000 target) | +14.0 [+11.6, +16.4] | +31.0 [+26.2, +35.6] | +13.6 [+11.5, +15.7] |
| C1 − STRICT | +5.9 [+4.2, +7.7] | +4.5 [+1.8, +7.3] | +2.1 [+1.4, +2.9] |
| C5 − ANSONLY (removing context) | +2.9 [+1.4, +4.5] | +6.0 [+3.2, +9.0] | +2.2 [+1.3, +3.1] |

Main conclusions: at equal size, composition dominates (SWAP and TGT4K reach about C5-level F1, and both intervals against C5 include zero); source data alone scale slowly; STRICT (no target labels at any stage) reaches 70.41 F1 with high variance, but it also has 2,000 fewer training examples and a source-only validation set, so its gap to C1 combines several factors.

### 13.4 Answer-only signal

TF-IDF logistic regression by input segment on the test set (F1 / MCC / AUROC):

| Input segment | C1 data | C5 data |
|---|---|---|
| Question only | 66.67 / 0.00 / 51.09 | 66.67 / 0.00 / 51.55 |
| Context only | 66.67 / 0.00 / 47.41 | 66.67 / 0.00 / 47.10 |
| Question + context + answer | 67.21 / 10.46 / 63.08 | 71.06 / 32.44 / 72.54 |
| Question + answer | 76.95 / 50.99 / 83.86 | 83.08 / 66.62 / 91.12 |
| Answer only | 81.42 / 62.24 / 87.73 | **86.23** / 72.60 / 93.84 |

An F1 of 66.67 with MCC 0 is what flagging every answer gives. Much of the benchmark signal lies in the answer text; removing the context from the retrained detector costs about 2.9 F1.

### 13.5 PPV at lower prevalence

Original-run projection (paper Eq. 2, sensitivity and specificity held fixed), PPV %: C5 95.1 / 82.8 / 68.2 / **50.3** at 50 / 20 / 10 / 5% prevalence; C1 72.9 / 40.2 / 23.0 / 12.4; DeepSeek-V3.2 11.0 and Llama-3.3-70B 27.2 at 5%.

Empirical resampling in the rerun (all 500 supported answers kept, 26/56/125 hallucinated sampled; 4.9% / 10.1% / 20.0% prevalence; precision %):

| Condition | FPR | 4.9% | 10.1% | 20.0% |
|---|---|---|---|---|
| C5 | 3.6 | 57.1 | 74.1 | 86.4 |
| SWAP | 5.5 | 46.7 | 65.3 | 80.8 |
| TGT4K | 5.5 | 46.6 | 65.3 | 80.7 |
| ANSONLY | 8.1 | 36.9 | 55.7 | 73.7 |
| STRICT | 23.7 | 13.8 | 25.5 | 43.3 |
| C1 | 31.3 | 12.0 | 22.5 | 39.5 |

The 57.1% (rerun, 4.9%) and 50.3% (original, 5%) values come from different runs and calculations and are not a direct replication.

---

## 14. Limitations and scope

- **One benchmark family.** Source and target share an upstream lineage. Target training/validation come from `pqa_artificial` and the test set from `pqa_labeled`, so part of the measured gap may reflect differences between those subsets. No independent benchmark, held-out clinical specialty or unseen generator was evaluated.
- **Target labels in every condition but STRICT.** Checkpoint selection, calibration and threshold selection use 4,000 target validation labels, so the benefit of the first 1,000 training examples assumes that validation budget and says nothing about the best allocation of a fixed label budget.
- **Over-flagging mechanism untested.** The paper's disjunctive-claims account assumes independent claim-level errors and could not be tested because every test answer is a single sentence.
- **Baseline comparability.** Fixed 0.5 thresholds and uncalibrated verbalised probabilities for the judges vs validation-selected thresholds and temperature scaling for the detector; single runs; one shared judge prompt; routed serving backends.
- **Statistics.** Three seeds, uncorrected p-values; seed SDs and bootstrap intervals measure different things and their overlap is not a significance test. Per-example predictions were not retained for the original adaptation runs, so detector-vs-baseline comparisons from the original runs are descriptive.
- **Test-set reuse.** The same 1,000-example test set is used throughout; further model development should use validation data and a new held-out test set.
- **Labels not re-annotated.** Labels are inherited from the source and benchmark pipelines. PubMedQA long answers (conclusions) may contain claims not stated in the supplied context, so some detector false positives may reflect label construction rather than detector error. Per-token grounding scores are not calibrated probabilities of token-level support and do not validate explanations.
- **Not clinical evidence.** No clinical records were used and no clinical-validation claim is made. Results are benchmark-relative and are not evidence of clinical readiness. Flags should prompt manual review of the supplied evidence, never act as an autonomous clinical authority.

---

## Further documentation

[Resuming interrupted runs](docs/resume.md) · [architecture](docs/architecture.md) · [data protocol](docs/data_protocol.md) · [baseline protocol](docs/baseline_protocol.md) · [shared-answer protocol](docs/shared_answer_protocol.md) · [revision protocol](docs/revision_protocol.md) · [API sources](docs/api_sources.md) · [2.2 changes](docs/changes_2_2.md) · [validation report](docs/validation_report.md) · [submission guide](SUBMISSION_GUIDE.md) · [prompts](prompts/README.md)

## Data availability and licence

Data derive from the publicly available PubMedQA and MedHallu datasets (both MIT-licensed). Derived files are released under the upstream MIT terms, with attribution, upstream versions and a record of every transformation step. The transformed files lack record-level generation lineage and embedded licence/provenance fields; the repository documents the upstream sources and transformation steps but cannot recover the missing lineage.
