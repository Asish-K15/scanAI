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

## Status

Model: FROZEN

ONNX export: VERIFIED

Independent test: COMPLETE

Dataset provenance: PENDING

Production integration: IN PROGRESS
