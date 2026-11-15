# External evaluation sets (not shipped)

Put here, then score with `clinhallu eval-only` (frozen checkpoint, frozen calibration):

* `halubench_pubmedqa.jsonl` -- from `python scripts/prepare_external.py --source <HaluBench file>`
* `unseen_generator.jsonl`   -- from `python scripts/build_unseen_generator_set.py ...`
* `heldout_abstracts.jsonl`  -- your ~500 PubMed question/abstract pairs (input to the builder)

Each builder also writes a `.card.json` (size, balance, label definition, hashes) and a
`.audit.csv` for the manual label check. The worker skips eval jobs whose dataset is missing.
