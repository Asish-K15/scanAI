# Cat Eye Integration Handoff

> **DOCUMENT TYPE:** CURRENT REPOSITORY VERIFICATION
> **STATUS:** VERIFIED FACTUAL BASELINE
> **SCOPE:** FROZEN CAT EYE INFERENCE SERVICE & API INTEGRATION
> **LATEST INTEGRATION COMMIT:** `0bbe095b9449870414014040598dcb1d310ced58`

---

## 1. Overview & Verification Context

This document establishes the verified baseline for the frozen Cat Eye disease screening model integrated into ScanAI. All specifications herein are derived directly from active repository code (`app/services/cat_eye.py`, `app/routers/predict.py`), the verified checkpoint metadata (`best_checkpoint.pth`), and unit tests (`tests/test_cat_eye.py`).

No historical training records or unverified provenance are assumed beyond what is mathematically and structurally verified within the repository.

---

## 2. Checkpoint & Artifact Verification

- **Model Identifier:** `scanai_cat_eye_efficientnet_b0_v1`
- **Model Version:** `v1.0.0`
- **Architecture:** `EfficientNet-B0` (classifier replaced with 6 output logits)
- **Engine:** PyTorch (`torchvision.models.efficientnet_b0`)
- **Checkpoint Location:** `dataset_manager/training/outputs/cat_eye/efficientnet_b0/best_checkpoint.pth`
- **Checkpoint File Size:** `16,363,295 bytes`
- **Checkpoint SHA-256:** `997576aa344eeec8f03c30d4b1d501850a89921e204a6b16455e147d5bd8ab7b`
- **Immutability:** Checkpoint is frozen; verified on load against expected SHA-256.

---

## 3. Checkpoint Internal Metadata

Verified directly from the checkpoint payload:

- **Best Epoch:** `11`
- **Validation Macro-F1:** `0.6808088473948488` (~`68.08%`)
- **Optimizer:** `AdamW`
- **Learning Rates:** Backbone LR = `5e-05`, Classifier LR = `5e-04`
- **Weight Decay:** `0.0001`
- **Scheduler:** `CosineAnnealingLR`
- **Batch Size:** `16`
- **Total Training Epochs:** `18`
- **Random Seed:** `42`
- **Loss Function:** `CrossEntropyLoss (class-weighted)`
- **Class Weights:**
  - `cataracts`: `1.4960317611694336`
  - `conjunctivitis`: `0.6684397459030151`
  - `cornealulcer`: `2.026881694793701`
  - `glaucoma`: `2.992063522338867`
  - `healthy`: `0.41337719559669495`
  - `uveitis`: `1.6981981992721558`
- **Selection Metric:** `validation macro-F1`

---

## 4. Exact Class Mapping

The model outputs exactly six classes with fixed indices (0-5):

| Index | Class Name | Description |
|:---:|:---|:---|
| 0 | `cataracts` | Cataract condition |
| 1 | `conjunctivitis` | Conjunctival inflammation |
| 2 | `cornealulcer` | Corneal ulceration |
| 3 | `glaucoma` | Glaucoma condition |
| 4 | `healthy` | Normal / Healthy eye |
| 5 | `uveitis` | Uveal tract inflammation |

---

## 5. Input Preprocessing Pipeline

- **Color Space:** RGB (`image.convert("RGB")`)
- **Spatial Dimensions:** `224 x 224` pixels (`transforms.Resize((224, 224))`)
- **Tensor Conversion:** `transforms.ToTensor()` (scales values to `[0.0, 1.0]`)
- **Normalization:** Standard ImageNet normalization:
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard Deviation: `[0.229, 0.224, 0.225]`

---

## 6. Crop Protection Policy

- **Minimum Image Dimension:** `32 x 32` pixels (`MIN_IMAGE_DIM = 32`).
- **Pre-inference Gate:** Any image or crop with `width < 32` or `height < 32` raises `CatEyeInvalidCropError` **before** model inference or tensor allocation occurs.
- **API Behavior:** Translated into HTTP `400 Bad Request` at `/api/predict`.

---

## 7. Uncertainty and Confidence Policies

- **Confidence Score:** Derived from `softmax(logits)` corresponding to the top predicted class.
- **Uncertainty Threshold:**
  - If `confidence < 0.40`: `is_uncertain = True`, `uncertain = True`.
  - If `confidence >= 0.40`: `is_uncertain = False`, `uncertain = False`.
- **Confidence Levels:**
  - `confidence >= 0.80`: `"high"`
  - `0.60 <= confidence < 0.80`: `"moderate"`
  - `confidence < 0.60`: `"low"`

---

## 8. Integration & Pipeline Flow

The production inference path is exposed via `POST /api/predict`:

1. **HTTP Form Validation:** Image file type, `species`, and `body_area`.
2. **Species Routing:** `route_species(species)` validates selected species.
3. **Animalness Gate Validation:**
   - Image and selected species are validated by `AnimalnessGateService`.
   - If gate decision is `REJECT` (including `SPECIES_MISMATCH` or non-animal rejection), execution halts immediately and returns gate payload. No Cat Eye inference runs.
   - If gate decision is `UNCERTAIN`, execution halts immediately. No Cat Eye inference runs.
   - Only `ACCEPT` proceeds downstream.
4. **Body Area Routing:** `route_body_area(body_area)` confirms `"eye"`.
5. **Disease Model Execution:**
   - Routes `species == "cat"` and `body_area == "eye"` to `get_cat_eye_model().predict(pil_image)`.
6. **Recommendation Generation:** Output passes to `build_recommendation()` to construct standard response envelope.

---

## 9. Clinical Boundaries & Fusion Guarantees

- **Screening-Only:** `screening_only = True` in all inference outputs.
- **No Diagnostic or Treatment Derivation:** Cat Eye condition, confidence score, and confidence level are strictly prohibited from deriving clinical severity, clinical urgency, or veterinary treatments.
- **Default Urgency Behavior:** In the absence of independently validated clinical evidence, `/api/predict` returns:
  - `severity: null`
  - `urgency: null`
  - `evidence_status: "insufficient_evidence"`
  - `recommendation: "insufficient evidence / urgency undefined"`
- High model confidence (e.g. `0.999`) never promotes urgency to `Emergency` or `Routine`.

---

## 10. Documentation Gap Assessment

- `dataset_manager/expansion/cat_eye_audit/final/cat_eye_model_manifest.json`: **MISSING** (The directory path `dataset_manager/expansion` does not exist in the repository; no expansion audit manifest JSON was generated).
- `CAT_EYE_INTEGRATION_HANDOFF.md`: **CREATED (This document)** based exclusively on verified repository artifacts and code.
