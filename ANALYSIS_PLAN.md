# ANALYSIS_PLAN (commit BEFORE any new result exists)

Fill in, then `git commit`. Do not edit after the first new `DONE`.

1. Primary comparisons (paired grouped bootstrap, 3 seeds pooled, 95% CI): C5-SWAP, SWAP-C1, C5-TGT4K, C1-STRICT, C1-S5K, S5K-S2K.
2. Primary metrics: F1 at the frozen validation threshold; secondary MCC, AUROC, AUPRC, ECE.
3. What each outcome means: see the decision table in docs/revision_protocol.md section 6.
4. Evidence dependence: C1/C5 normal vs blank vs shuffled; ANSONLY vs C5; TF-IDF and answer-length baselines on answer text only.
5. External: frozen C1/C5 (no retuning) on the external set and the unseen-generator set; report AUROC and the frozen-threshold F1.
6. Secondary: claim-count vs FP rate; low-prevalence resampling (26/56/125 hallucinated, 100 repeats); judge calibration (validation-fitted Platt + threshold).
7. Exclusions / stopping rules: <write here>.
8. Anything chosen using test results must be flagged as exploratory.
