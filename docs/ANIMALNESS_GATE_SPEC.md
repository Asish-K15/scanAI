# ScanAI — Animalness Gate Specification & Integration Guide

## 1. Overview & Purpose

The **Animalness Gate** is a critical validation layer in ScanAI / PetVision AI positioned directly ahead of specialized disease screening models (Dog Eye and Skin models). Its purpose is to independently verify that:
1. The uploaded image actually contains a supported animal.
2. The detected species matches the user's selected species (`dog`, `cat`, `cattle`).
3. Non-animal objects, human images, and corrupted files are halted before reaching disease models.

> **Status Notice:**
> The Animalness Gate is integrated for engineering and test execution. Per the PRD, the checkpoint is a diagnostic candidate and has **not** yet been given final production approval for model performance.
> - New hard-negative human set: 299/300 rejected (1 false accept).
> - Independent 30-image human challenge: 21/30 rejected (9 false accepts).
> Model evaluation and performance optimization remain an active, separate track.

---

## 2. Architecture & Checkpoints

The Animalness Gate utilizes two frozen EfficientNet-B0 classifiers in tandem:

| Component | Model Backbone | Checkpoint | Output Classes | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Animalness Detector** | EfficientNet-B0 (`SCANAI-ANIMALNESS-GATE-V1`) | `animalness_gate/outputs/best_animalness_gate.pth` | `animal`, `not_animal` | Binary detection of animal vs non-animal / human |
| **Species Validator** | EfficientNet-B0 | `animal_validator/outputs/best_species_validator.pth` | `dog`, `cat`, `cattle` | Multi-class classification among supported species |

Both checkpoints are **strictly frozen** and must not be modified or overwritten.

### Image Preprocessing Transform
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

---

## 3. Required Gate Contract

All gate evaluations return a structured response conforming to `AnimalnessGateResponse`:

```json
{
  "decision": "ACCEPT",
  "predicted_species": "dog",
  "species_confidence": 0.9991,
  "animal_probability": 0.9996,
  "model_name": "EfficientNet-B0",
  "model_version": "SCANAI-ANIMALNESS-GATE-V1",
  "reason_code": "SUPPORTED_ANIMAL_DETECTED",
  "error_code": null
}
```

### Allowed Values

- **`decision`**:
  - `ACCEPT`: Image is a confirmed supported animal matching the selected species.
  - `REJECT`: Image is non-animal, human, species mismatch, unsupported, or invalid.
  - `UNCERTAIN`: Image confidence falls within configured uncertainty boundaries.

- **`reason_code`**:
  - `SUPPORTED_ANIMAL_DETECTED`
  - `HUMAN_DETECTED`
  - `NON_ANIMAL_DETECTED`
  - `LOW_CONFIDENCE`
  - `SPECIES_MISMATCH`
  - `UNSUPPORTED_SPECIES`
  - `INVALID_IMAGE`
  - `MODEL_ERROR`

---

## 4. Species Mismatch Rule (Never Silently Reroute)

If the user selects one species (e.g. `dog`) and the Animalness Gate detects a different species (e.g. `cat`), the request is **strictly rejected**:

```json
{
  "decision": "REJECT",
  "predicted_species": "cat",
  "species_confidence": 0.9952,
  "animal_probability": 0.9930,
  "model_name": "EfficientNet-B0",
  "model_version": "SCANAI-ANIMALNESS-GATE-V1",
  "reason_code": "SPECIES_MISMATCH",
  "error_code": null
}
```

The system **never** silently reroutes the image to the Cat model.

---

## 5. Uncertainty & Configurable Thresholds

Per PRD Section 9:
- No arbitrary production confidence threshold is hard-coded into the service, and **no implicit fallback threshold (such as 0.50) is assumed**.
- When `animal_threshold` is unconfigured (`animal_threshold = None`), the service returns `decision: "UNCERTAIN"` and `reason_code: "LOW_CONFIDENCE"` to prevent unapproved production acceptance or rejection.
- When an explicit threshold is configured (`animal_threshold` is a float):
  - Images below the threshold return `decision: "REJECT"` and `reason_code: "NON_ANIMAL_DETECTED"`.
  - `HUMAN_DETECTED` is preserved in the schema contract for future independently established human detectors, but the binary gate does not pretend to identify humans specifically.
  - Images at or above the threshold proceed to species validation.
- Configurable parameters in `AnimalnessGateConfig`:
  - `animal_threshold`: Explicit decision threshold.
  - `uncertainty_min` / `uncertainty_max`: Optional uncertainty interval mapping to `UNCERTAIN` / `LOW_CONFIDENCE`.
  - `species_confidence_threshold`: Optional species confidence gate.

This architecture ensures production thresholds can be configured later without requiring code modifications.

---

## 6. Integration Flow in `/api/predict`

```text
POST /api/predict
        │
        ├── 1. Validate image format & read bytes
        │
        ├── 2. Validate selected species in {"dog", "cat", "cattle"}
        │
        ├── 3. Animalness Gate validation
        │       │
        │       ├── REJECT → return structured gate response immediately
        │       │
        │       ├── UNCERTAIN → return structured gate response immediately
        │       │
        │       └── ACCEPT
        │
        ├── 4. Validate body area ("eye", "skin")
        │
        ├── 5. Route to frozen disease models (Dog Eye / Skin ONNX)
        │
        └── 6. Build approved recommendation schema
```

When the Animalness Gate rejects or flags an image as uncertain, disease models are **never called**.

---

## 7. Protected Datasets & Models

The following artifacts are frozen and protected against modifications:
1. `animal_validator/rejection_samples/human/` (30 external human challenge images).
2. `animalness_gate/data/test/` (untouched animal test set).
3. `dataset_manager/training/outputs/phase4_final/scanai_skin_phase4b.onnx` (Skin Phase 4B screening model).
4. `animalness_gate/outputs/best_animalness_gate.pth` (Animalness checkpoint).
5. `animal_validator/outputs/best_species_validator.pth` (Species validator checkpoint).

---

## 8. Verification Commands

Run unit tests:
```bash
.venv\Scripts\python -m unittest discover -s tests -p "test_animalness_gate_service.py"
```

Run integration tests:
```bash
.venv\Scripts\python -m unittest discover -s tests -p "test_animalness_gate_integration.py"
```

Run full test suite:
```bash
.venv\Scripts\python -m unittest discover -s tests
```
