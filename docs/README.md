# docs/

Living documentation for the report, presentation, and any future write-up.
Update these as you go — don't reconstruct them at the end.

| File | Purpose | Status |
|---|---|---|
| `dataset_sources.csv` | Every dataset used, with license and counts | Pre-filled with what's been vetted so far — add rows as you add datasets |
| `licence_log.csv` | Mirror of `dataset_manager/licenses_template.csv` | Keep both in sync, or just symlink in your local clone |
| `experiment_log.csv` | One row per training run | Auto-appended by `01_train_baseline.ipynb` — don't edit by hand |
| `architecture.png` | System architecture diagram | **Not yet created** — build this once the pipeline is stable, from the diagram in the original project plan doc |
| `pipeline.png` | Dataset manager pipeline diagram | **Not yet created** — the ASCII diagram from earlier conversations is a good starting point |
| `results/` | Screenshots of confusion matrices, ROC curves, Grad-CAM examples for the report | Populate after each significant training run — pull from `outputs/` |

## Why this exists

Writing a report or SIH submission from memory at the end is where teams lose
the most avoidable points — dataset provenance gets fuzzy, exact metrics get
misremembered, and the "why we made this choice" reasoning gets lost. Filling
this in as you go turns your report into an assembly job instead of a
reconstruction job.
