# external_test/

**Rule: nothing in this folder ever passes through `dataset_manager/`'s
pipeline.** No download.py, no dedup, no augmentation, no training.

This is your real-world generalization check — photos that never touched
Roboflow, Kaggle, or your own training/val/test splits.

## What goes here

- Your own pets' photos
- Friends'/family's pet photos (with consent, per the earlier consent note)
- Farm/cattle photos if you get vet-clinic or shelter access
- Ideally, a few photos actively taken with the same kind of camera and
  conditions a real user would use — not curated, well-lit "dataset-quality" shots

## Structure

```
external_test/
├── skin__ringworm/
├── skin__mange/
├── parasites__tick_infestation/
├── ...  (same unified class names as taxonomy.json)
└── unlabeled/          # photos you're not fully sure of the label for —
                          useful for a "does the model's confidence match
                          my uncertainty" sanity check, not for accuracy scoring
```

Use the exact same class folder names as `dataset_manager/taxonomy.json`
so the evaluation cell in the notebook can load it directly.

## When to use it

**Only evaluate on this after the model is otherwise finalized** — this
folder is not for iterative tuning. If you keep tweaking the model based on
external_test performance, it stops being an honest holdout and starts
being a second validation set. Use `splits/val` and `splits/test` for all
iteration; come here once, near the end, for your final generalization number.
