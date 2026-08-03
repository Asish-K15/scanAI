# ScanAI (VetVision AI) — Master Plan & Day 1 Start Guide

*One consolidated reference. Everything below reflects decisions already made — this is the starting line, not a new discussion.*

---

## The project, in one line

AI-powered mobile app that detects visible, photo-diagnosable animal health issues — skin disease, parasites, wounds, eye/ear conditions, body condition — from a smartphone photo, across dogs, cats, and cattle, with confidence-graded, honest recommendations rather than overclaimed diagnoses.

---

## Locked scope (v1)

**Species:** Dogs and cats (primary), cattle (secondary, shares architecture)

**Features — build these:**
| Feature | Confidence tier |
|---|---|
| Skin disease (ringworm, mange, hotspot, dermatitis, etc.) | Flagship — strongest data |
| Parasites (ticks, fleas, ear mites) | Solid, thinner data than skin |
| Eye conditions (pink eye, cherry eye, eye infection) | Solid |
| External ear signs (redness, mites) | Solid — explicitly NOT internal otitis diagnosis |
| Wound severity (mild/moderate/severe) | Self-collected, secondary |
| Body condition score (under/ideal/overweight) | Self-collected, secondary |
| Species detection (dog/cat/cattle) | Easy, near-solved, nice UX touch |

**Explicitly cut / deferred:**
- Hoof/lameness/gait — genuinely a video problem, not a photo problem
- Internal ear infection (otitis media), distemper, parvovirus, feline leukemia, mastitis, dental disease — not photo-diagnosable, would undermine credibility
- Breed detection, age estimation, coat-type classification, segmentation masks — each its own research project, out of scope for this timeline

---

## Tech stack

| Layer | Choice |
|---|---|
| Mobile app | Flutter |
| Backend | FastAPI |
| Vision model | EfficientNet-B0 (transfer learning), ConvNeXt-Tiny/MobileNetV3 as comparisons |
| Database | PostgreSQL + Firebase Storage |
| Dashboard | Tableau / Power BI |
| Explainability | Grad-CAM |
| Deployment | Docker + cloud (Render/Railway for MVP) |
| Model export | TensorFlow SavedModel → TFLite (on-device inference) |

---

## What's already built and ready to use

```
ScanAI/
├── dataset_manager/         ✅ built — download → extract → taxonomy map →
│                                validate → dedup → split → stats pipeline
│   ├── taxonomy.json         ✅ your frozen class taxonomy
│   ├── licenses_template.csv ✅ start filling this in as you download
│   └── (run this first, against your real datasets)
├── notebooks/
│   └── 01_train_baseline.ipynb  ✅ built — full 15-section training notebook
│                                     + experiment logging + external holdout eval
├── docs/                     ✅ scaffolded — dataset_sources.csv, licence_log.csv,
│                                 experiment_log.csv, README explaining each
├── external_test/            ✅ scaffolded — your real-world holdout, empty for now
├── models/, outputs/          ✅ scaffolded, will populate once you train
├── app/, dashboard/            — not started (comes after a working baseline model)
```

---

## Day 1 checklist — do these in order

**1. Set up accounts & keys (30 min)**
- [ ] Kaggle account + API token (`~/.kaggle/kaggle.json`)
- [ ] Roboflow account + API key (`ROBOFLOW_API_KEY` env var)
- [ ] GitHub repo created, both of you with push access
- [ ] Google Colab access confirmed (free GPU runtime works)

**2. Vet/shelter outreach — start today, longest lead time**
- [ ] Email 1-2 local vet clinics or shelters about photo access for body condition / wound data and a possible validation spot-check later

**3. Fill in `dataset_manager/download.py`**
- [ ] Add the Kaggle slugs and Roboflow workspace/project/version you've already vetted across this conversation (the Kendys skin-disease set, the mange-search parasite/eye/ear sets, the cattle multi-disease set, etc.)
- [ ] Check each against the 4-point list first: class list, count, license, sample-quality spot check

**4. Run the pipeline**
```bash
cd dataset_manager
python download.py
python main.py
```
- [ ] Check console output for `_UNMAPPED` classes — add missing aliases to `taxonomy.json`, re-run `taxonomy_mapper.py`
- [ ] Check `metadata/dataset_inventory.csv` — note which classes are below the 200-image minimum-viable threshold

**5. Log what you used**
- [ ] Fill in `docs/dataset_sources.csv` and `docs/licence_log.csv` as you go, not after

**6. Run the training notebook**
- [ ] Open `notebooks/01_train_baseline.ipynb` in Colab (GPU runtime)
- [ ] Update `PROJECT_ROOT` to point at your Drive-mounted or uploaded folder
- [ ] Run top to bottom
- [ ] Check `outputs/metrics/classification_report.csv` for your 5 weakest classes

**7. Decide next step based on real results — not before**
- If baseline accuracy is reasonable: move to `02_error_analysis.ipynb` (build this once you have real weak classes to investigate)
- If specific classes are clearly starved of data: targeted self-collection for exactly those classes, not a broad re-search

---

## The 6-week team split (recap)

Both partners 50/50, pairing on the hardest integration points (fusion/urgency logic, upload→inference→report round trip). Full week-by-week breakdown is in the earlier project-plan conversation — the short version:

- **Week 1:** Setup + dataset pipeline (what you're doing right now)
- **Week 2:** Species detector + skin-disease baseline model
- **Week 3:** Wound severity + backend integration + fusion/urgency layer
- **Week 4:** Core app screens (camera → report)
- **Week 5:** Body condition model + history/profile screens
- **Week 6:** Validation, dashboard, demo prep

---

## The one rule to hold onto from here

**Decisions from this point forward are driven by what the trained model actually gets wrong — not by more planning.** Every dataset, every architecture choice, every feature has already been reasoned through in this conversation. The next new information you get should come from `outputs/metrics/classification_report.csv`, not from another round of "what else is needed."
