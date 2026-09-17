# Cat Eye Dataset Audit

**Project:** ScanAI
**Role:** Partner B — Dataset and Model Pipeline
**Phase:** Cat Eye Phase 1 — Dataset Discovery and Quality Audit
**Status:** Pending
## 1. Dataset Overview

### Source

- Dataset: Cat Eye Disease Detection
- Platform: Roboflow
- Dataset URL: https://universe.roboflow.com/student-s-workspace/cat-eye-disease-detection-fe2ca
- License: MIT
- Dataset export date: August 22, 2026
- Annotation format: COCO

### Classes

1. Cataracts
2. Conjunctivitis
3. Corneal_Ulcer
4. Glaucoma
5. Healthy
6. Uveitis

### Dataset Counts

| Split | Images | Annotations |
|---|---:|---:|
| Train | 281 | 356 |
| Valid | 80 | 99 |
| Test | 40 | 49 |
| Total | 401 | 504 |

## 2. Initial Annotation Audit

### 2.1 Bounding-Box Inconsistency

The audit identified two very small bounding-box cases in the test split.

Additional boundary checks identified two minor floating-point boundary violations, likely caused by annotation rounding.

Observed findings:

- 2 very small bounding-box cases in the test split
- 2 minor boundary rounding cases
- No invalid box dimensions
- Original annotations preserved

The small bounding boxes require manual annotation review.
No automatic annotation correction has been performed.

### 2.2 Multi-Annotation Images

- Training split: 75 images with multiple annotations
- Valid split: 19 images with multiple annotations
- Test split: 9 images with multiple annotations

Multiple annotations may represent separate eyes, multiple regions, repeated same-class annotations, or mixed labels. Their exact meaning requires further review.

### 2.3 Healthy + Disease Annotations

A total of 14 images contain both Healthy and disease labels.

| Split | Mixed-Label Images |
|---|---:|
| Train | 8 |
| Valid | 4 |
| Test | 2 |
| Total | 14 |

Label combinations:
- Cataracts + Healthy: 13 images
- Conjunctivitis + Healthy: 1 image

Visual inspection of the mixed-label samples was completed.
No obvious annotation placement errors were identified.
Original annotations remain unchanged.

Clinical label correctness cannot be confirmed through visual inspection alone.
### 2.4 Potential Class Overlap

Potential sources of visual confusion include:

- Cataracts and glaucoma
- Corneal ulcer and other corneal abnormalities
- Uveitis and other inflammatory eye conditions
- Healthy regions alongside diseased regions

Visual inspection alone cannot establish clinical label correctness.

## 3. Data Quality Concerns

- [ ] Mixed `Healthy + Disease` images
- [ ] Repeated same-class annotations
- [ ] Duplicate or overlapping bounding boxes
- [ ] Extremely small bounding boxes
- [ ] Extremely large bounding boxes
- [ ] Train/test class consistency
- [ ] Image-level versus eye-level label consistency
- [ ] Dataset provenance and license documentation

## 4. Proposed Cleaning Rules

No original annotation has been deleted or modified.

1. Preserve the original dataset as an immutable reference.
2. Record questionable annotations in an audit report.
3. Do not automatically remove mixed-label images.
4. Check whether multiple annotations represent separate eyes.
5. Investigate duplicate and overlapping bounding boxes.
6. Review unusually small and large bounding boxes.
7. Verify class consistency between splits.
8. Create a cleaned copy only after policy approval.
9. Maintain an audit log for all modifications.

## 5. Model Approach Assessment

| Approach | Requirement | Current Status |
|---|---|---|
| Image classification | Reliable image-level or eye-level labels | Not approved |
| Object detection | Consistent bounding boxes and object definitions | Not approved |
| Eye-crop classification | Validated eye-level crops and labels | Proposal only |

Eye-crop classification remains a proposal until separate-eye annotations and mixed-label samples are verified.

## 6. Visual and Clinical Validation Limitations

- Image inspection cannot confirm a clinical diagnosis.
- Visible abnormalities do not prove the assigned disease label.
- Clinical confirmation requires qualified veterinary assessment or reliable expert annotation.
- The model must not be presented as a definitive veterinary diagnostic system.

## 7. Training Readiness Decision

**Status: PENDING**

Cat Eye model training is not approved yet.

Training readiness depends on:

1. Completion of the annotation quality audit.
2. Review of mixed-label and multi-annotation images.
3. Agreement on annotation and cleaning policy.
4. Verification of class consistency.
5. Selection and approval of the model formulation.
6. Documentation of dataset provenance and limitations.

## 8. Audit History

| Date | Action | Result |
|---|---|---|
| 2026-09-17 | Dataset discovery | Completed |
| 2026-09-17 | Full train/valid/test annotation audit | Completed |
| 2026-09-17 | Representative visual inspection | Completed |
| 2026-09-17 | Training-readiness decision | Pending |
| 2026-09-17 | Duplicate and reference audits | Completed |
| 2026-09-17 | Small bounding-box review | Flagged for manual review |

