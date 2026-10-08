# ScanAI Dog Eye Disease Model

## Model

EfficientNet-B0 fine-tuned for dog eye disease classification.

## Classes

| Label | Class |
|---|---|
| 0 | conjunctivitis |
| 1 | entropion |

## Dataset

Total images: 1,344

| Split | Conjunctivitis | Entropion | Total |
|---|---:|---:|---:|
| Train | 481 | 484 | 965 |
| Validation | 145 | 116 | 261 |
| Test | 68 | 50 | 118 |
| Total | 694 | 650 | 1,344 |

Dataset provenance: TBD — original source metadata has not yet been recovered.

License: TBD — pending provenance verification.

## Training

Architecture: EfficientNet-B0

Fine-tuning was performed using the training and validation sets.

The independent test set was not used during model selection or fine-tuning.

Best validation accuracy:

96.55%

Best epoch:

7

## Independent Test Results

Test samples: 118

Accuracy: 94.92%

Entropion precision: 92.31%

Entropion recall: 96.00%

Entropion F1: 94.12%

ROC-AUC: 97.56%

## Confusion Matrix

| Actual / Predicted | Conjunctivitis | Entropion |
|---|---:|---:|
| Conjunctivitis | 64 | 4 |
| Entropion | 2 | 48 |

## Model Selection

The fine-tuned model was selected over the targeted-augmentation experiment.

Both achieved 94.92% independent test accuracy.

The fine-tuned model had higher entropion recall:

96.00% vs 94.00%.

For ScanAI's triage-oriented design, reducing missed disease cases is important.

## Deployment

The PyTorch checkpoint was exported to ONNX.

ONNX model:

`scanai_dog_eye_efficientnet_b0.onnx`

ONNX structural validation: PASS.

PyTorch CPU vs ONNX CPU consistency: PASS.

## Production Preprocessing

Input:

RGB image

Resize:

224 × 224

Normalization:

Mean:

[0.485, 0.456, 0.406]

Standard deviation:

[0.229, 0.224, 0.225]

## Output

The model produces two logits:

0 → conjunctivitis

1 → entropion

Softmax probabilities should be used to calculate confidence.

## Important Scope

This model is an image-based classification component of ScanAI.

It should not be presented as a complete veterinary diagnostic system.

Image-based predictions should be treated as screening/triage assistance and should not replace professional veterinary assessment.

## Provenance, Integrity, and Licensing Audit

### Checkpoint Integrity

- Production ONNX path: `dog_eye_model/outputs/onnx/scanai_dog_eye_efficientnet_b0.onnx`
- SHA-256: `1c03ec5a38ce37b5b654b43b869d685a77e0d1539611be254d50e40b138611dc`
- Size: 16,030,443 bytes
- The archived ONNX artifact (`dog_eye_model/scanai_dog_eye_v1_onnx_artifact.zip`) matches the production ONNX file.

### Dataset Identity

- Master manifest: `dataset_manager/metadata/dog_eye_disease_master_manifest.csv`
- Total images: 1,344
- Split breakdown: 965 train, 261 valid, 118 test
- Classes: conjunctivitis and entropion

### Provenance

- Repository source registry (`docs/dataset_sources.csv`) identifies:
  - Dataset: "Dog eye problems detection"
  - Source: Roboflow
  - Identifier: `jonathan-chandra/dog-eye-problems-detection`
- Source registry status remains pending.
- The repository does not contain sufficient evidence to prove that the locally downloaded "Dog Eye Problems Detection-Forked on 8-22-2026" archive is definitively the same Roboflow export/version.
- Therefore provenance is incomplete.

### Licensing

- Model card and source registry currently indicate TBD / VERIFY.
- No verified Dog Eye dataset license is present in the repository.
- Licensing remains unresolved and unverified; no license is inferred or assigned.

### Dataset Integrity

- Master manifest audit demonstrates:
  - 1,344 unique filenames
  - 1,344 unique base image IDs
  - 1,344 unique UUIDs
  - Zero split overlap
  - Zero cross-class overlap
- No demonstrated train/test leakage was found in the audited master manifest.

### Dataset Quality Limitations

- Documented blurriness findings remain recorded as dataset-quality limitations.
- Data originated from object-detection annotations and was adapted for classification; these limitations are acknowledged and not eliminated.

### Production Scope

- Dog Eye remains `screening_only`.
- Confidence thresholds are engineering thresholds, not clinical severity mappings.
- Existing production behavior, preprocessing, and routing remain unchanged.

## Status

Model: FROZEN

ONNX export: VERIFIED

Independent test: COMPLETE

Dataset provenance: PENDING

Production integration: IN PROGRESS
