# ScanAI / VetVision AI — Project Root

```
ScanAI/
├── dataset_manager/     ← download, clean, dedupe, split (run this first)
├── notebooks/
│   └── 01_train_baseline.ipynb   ← START HERE for modeling
├── models/                ← trained model files land here
├── outputs/
│   ├── confusion_matrix/
│   ├── gradcam/
│   ├── reports/           ← training curves, ROC curves, class distribution charts
│   └── metrics/            ← classification_report.csv, run_metadata.json
├── app/                     ← Flutter app (not started yet)
└── dashboard/                ← Tableau/Power BI dashboard (not started yet)
```

## Workflow

1. **`dataset_manager/`** — follow its own README. End state: `dataset_manager/splits/{train,val,test}/` populated and leakage-safe.
2. **`notebooks/01_train_baseline.ipynb`** — run top to bottom in Google Colab (GPU runtime). Produces:
   - A trained model in `models/`
   - Training curves, confusion matrix, classification report, ROC curves in `outputs/`
   - `outputs/metrics/run_metadata.json` — your report's evaluation numbers, generated automatically each run
3. **Next notebooks (not yet built — build these once 01 gives you a working baseline):**
   - `02_error_analysis.ipynb` — misclassified images, lowest-confidence predictions, per-class breakdown. Use `outputs/metrics/classification_report.csv`'s weakest classes as your starting point.
   - `03_gradcam.ipynb` — heatmap explainability overlays for your report/demo
   - `04_export_model.ipynb` — TFLite conversion + on-device inference sanity check
   - `05_flutter_testing.ipynb` — confirms the exported model behaves identically outside the training environment before you wire it into the app

## Before you run 01_train_baseline.ipynb

Make sure `dataset_manager/splits/` actually exists and has real images in it —
run the full dataset_manager pipeline first:
```bash
cd dataset_manager
python download.py
python main.py
```
If `splits/` is empty or thin, the notebook will still run but your metrics
will be meaningless — check `dataset_manager/metadata/dataset_inventory.csv`
first to confirm you have at least the minimum-viable 200 images/class before
spending GPU time on a full training run.

## What "done" looks like for this notebook

- `outputs/metrics/run_metadata.json` has a test_accuracy you can defend
- `outputs/metrics/classification_report.csv` shows you exactly which classes
  are weak (almost certainly parasites/ear/eye — expected, not a bug)
- You know which 3-5 classes to prioritize for either more self-collected data
  or a closer look in `02_error_analysis.ipynb`
