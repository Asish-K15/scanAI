# ScanAI Fusion / Urgency Specification

## Status

Specification consolidated from reviewed and approved project decisions.

Implementation is permitted for explicitly approved v1 behavior. Additional implementation remains blocked for rules that are still undefined.

## Purpose

The Fusion / Urgency layer converts confidence-tagged vision-model outputs into application-level triage recommendations.

It is separate from the vision inference models and must remain independently versionable.

## Design Principles

1. Vision models provide probabilistic predictions.
2. Every vision prediction is treated as estimated from a photo.
3. Model confidence must not automatically be interpreted as disease severity or urgency.
4. Recommendation logic uses deterministic rule-based clinical logic together with model outputs.
5. ScanAI is a triage / first-look tool and is not a replacement for veterinary diagnosis.
6. Vision inference and recommendation logic remain separate components.

## Approved Urgency Categories

The application-level urgency categories are:

1. Routine
2. Soon
3. Urgent
4. Emergency

These categories define the application vocabulary only. Their assignment criteria are not necessarily fully defined.

## Currently Defined Severity

Wound severity is defined as:

- mild
- moderate
- severe

This represents severity grading rather than exact wound-type diagnosis.

Wound severity is a supported evidence category, but no automatic severity-to-urgency mapping has been approved.

## Approved Urgency Endpoints

### Routine

The documented low-risk endpoint is:

- minor/low-risk condition → Routine / non-urgent monitoring

No additional Routine mappings are established.

### Routine evidence input

For v1, the Fusion/Urgency engine may consume the following technical
evidence input:

`low_risk_evidence`

This input is an independently established upstream/deterministic
application assessment.

The Fusion/Urgency engine does not derive `low_risk_evidence` from:

- Dog Eye condition
- model confidence
- model confidence level
- clinical severity

When `low_risk_evidence = true`, the already-established upstream
assessment indicates that the case meets the project's minor/low-risk
criterion, and the Fusion/Urgency engine may assign:

`Routine`

When `low_risk_evidence = false` or the input is absent, the Routine
rule does not apply.

The criteria used by an upstream component to establish
`low_risk_evidence = true` are outside the Fusion/Urgency rules in this
specification unless separately approved.

`low_risk_evidence` must not override an independently established
Emergency rule.

This technical input does not define a new clinical classification,
condition mapping, severity mapping, confidence threshold, or
mathematical fusion rule.

### Emergency

The currently approved Emergency mapping is:

- severe deep-tissue laceration + active hemorrhage → Emergency

This is the only currently documented Emergency mapping.

Additional Emergency evidence remains undefined.

## Urgency Assignment Status

### Soon

Soon is an approved application-level urgency category for v1.

No specific observable condition is currently documented or approved as a Soon trigger.

For v1, the assignment criteria for Soon are intentionally undefined.

This is an explicit specification decision and must not be interpreted as permission to infer or invent a clinical Soon rule.

Therefore:

- Soon remains available as an application-level category.
- No specific condition or observable sign is currently mapped to Soon.
- No model confidence threshold determines Soon.
- No Dog Eye condition is mapped to Soon.
- No fallback rule automatically assigns Soon.

### Urgent

Urgent is an approved application-level urgency category for v1.

No specific observable condition is currently documented or approved as an Urgent trigger.

For v1, the assignment criteria for Urgent are intentionally undefined.

This is an explicit specification decision and must not be interpreted as permission to infer or invent a clinical Urgent rule.

Therefore:

- Urgent remains available as an application-level category.
- No specific condition or observable sign is currently mapped to Urgent.
- No model confidence threshold determines Urgent.
- No Dog Eye condition is mapped to Urgent.
- No fallback rule automatically assigns Urgent.

### Additional Emergency Evidence

No additional Emergency trigger is currently documented or approved.

Additional Emergency evidence remains undefined.

## Supported Evidence Categories

The Fusion / Urgency layer may receive or use the following evidence categories:

### Wound severity

Supported values:

- mild
- moderate
- severe

No automatic mapping from wound severity to urgency is currently approved.

### Deep-tissue laceration

Supported as evidence through the documented severe deep-tissue laceration example.

Its currently approved Emergency role is only in combination with active hemorrhage.

### Active hemorrhage

Supported as evidence through the documented high-urgency example.

No broader standalone Emergency mapping is currently approved.

### Detected condition / observable sign

Supported as an input/evidence category.

The documentation does not establish urgency mappings for individual conditions or signs.

No disease-specific urgency mapping is currently approved.

### Model confidence

Supported as contextual information.

Model confidence must remain separate from severity and urgency.

The Dog Eye confidence thresholds are engineering confidence thresholds only.

### Model uncertainty / insufficient evidence

Supported as an input to the recommendation layer.

Its effect on urgency is explicitly not defined.

## Uncertainty / Insufficient Evidence

The following behavior is approved:

- `uncertain = true` must not automatically map to Routine, Soon, Urgent, or Emergency.
- Insufficient evidence should be explicitly identified in the recommendation rather than silently treated as low severity.
- Any independently established urgency evidence should remain visible even when a model result is uncertain.
- If no urgency-relevant evidence is available, the final behavior is defined by the approved v1 insufficient-evidence behavior documented below.

Important distinctions:

- uncertainty ≠ low severity
- uncertainty ≠ high urgency

### Final Behavior When No Urgency-Relevant Evidence Exists

For v1, when no approved urgency-relevant evidence is available:

- `evidence_status` must be `"insufficient_evidence"`.
- `urgency` must remain undefined (`None` in the implementation).
- The recommendation must communicate `"insufficient evidence / urgency undefined"`.
- Existing model/output evidence must remain visible.
- The system must not fall back to `Routine`, `Soon`, `Urgent`, or `Emergency`.
- Model confidence and uncertainty must remain contextual and must not be converted into an urgency category.

This behavior preserves the distinction between:

- `Routine` as an approved low-risk endpoint
- `insufficient_evidence` as an explicit absence of sufficient approved evidence for an urgency determination

This is a v1 specification decision and is frozen for v1.

This decision does not introduce a new clinical rule, urgency trigger, confidence threshold, or mathematical fusion formula.

## Conflicting Signals

Conflicting evidence must be explicitly identified as a conflict in the recommendation layer.

The following constraints apply:

- The system must not silently discard one signal in favor of another.
- Conflicts must not be resolved using model confidence alone.
- Dog Eye confidence thresholds must not be used as conflict-resolution thresholds.
- Independently established urgency evidence should remain visible rather than being silently overridden.
- If a conflict cannot be resolved using an explicitly approved rule, the final urgency behavior remains undefined.

Therefore:

- conflicting signals ≠ choose highest confidence
- conflicting signals ≠ automatically Emergency
- conflicting signals ≠ automatically Routine

No mathematical conflict-resolution or fusion rule has been approved.

### Final Behavior for Unresolved Conflicts

For v1, when conflicting evidence cannot be resolved using an explicitly approved rule:

- The conflict must be explicitly represented in the recommendation/evidence output.
- All relevant conflicting evidence must remain visible.
- `urgency` must remain undefined (`None` in the implementation).
- The system must not fall back to `Routine`, `Soon`, `Urgent`, or `Emergency`.
- Model confidence must not be used to select a winning signal.
- Dog Eye confidence thresholds must not be used as conflict-resolution thresholds.
- No mathematical conflict-resolution or fusion rule may be introduced.

An unresolved conflict is not equivalent to low severity or Routine. It represents an insufficient basis for determining a final urgency.

This is a v1 specification decision and is frozen for v1.

## Dog Eye Confidence Thresholds

The Dog Eye model currently exposes:

- high: confidence >= 0.80
- moderate: confidence >= 0.60
- low: confidence < 0.60

These are engineering confidence levels only.

They are NOT clinical severity or urgency thresholds.

They must not be converted into urgency rules.

## Vision Model Output

The Fusion / Urgency layer may receive:

- species
- body area
- condition
- model confidence
- confidence level
- uncertainty flag
- model version
- screening-only status

These fields represent model/application evidence inputs. The exact final multi-model Fusion contract remains to be finalized.

## Dog Eye Severity / Urgency Mapping

For v1, the current project requirements do not provide sufficient evidence for a direct Dog Eye -> severity mapping or Dog Eye -> urgency mapping.

Therefore:

- Dog Eye -> severity remains intentionally undefined for v1.
- Dog Eye -> urgency remains intentionally undefined for v1.
- Dog Eye model outputs remain contextual evidence/model outputs.
- Dog Eye confidence remains engineering confidence information only.
- The Dog Eye confidence thresholds (`0.60 / 0.80`) must not automatically determine severity or urgency.

No disease-specific clinical mapping may be introduced without explicit, evidence-supported project approval.

## Explicitly Undefined

The following remain undefined and must not be invented:

- Soon assignment criteria
- Urgent assignment criteria
- Additional Emergency evidence
- Final behavior when no urgency-relevant evidence exists
- Mathematical fusion formula
- Dog Eye severity mapping
- Dog Eye urgency mapping
- Rules combining confidence, condition, and severity
- General conflict-resolution formula

## Prohibited Assumptions

The following must not be introduced without explicit project approval:

- confidence -> urgency conversion
- Dog Eye confidence threshold -> clinical severity
- Dog Eye confidence threshold -> clinical urgency
- disease-specific Dog Eye urgency mappings
- unsupported disease-specific severity mappings
- mathematical fusion formulas presented as previously agreed requirements
- unsupported clinical rules presented as existing ScanAI requirements

## Mathematical Fusion

Mathematical fusion is intentionally undefined for v1.

The current project requirements do not provide sufficient evidence that a mathematical fusion formula is required.

Therefore, v1 must not introduce:

- weighted confidence scores
- confidence -> urgency conversion
- severity x confidence formulas
- highest-confidence conflict resolution
- arbitrary numerical thresholds

Model confidence remains separate from urgency assignment unless an explicit, evidence-supported rule is approved.

The initial v1 implementation should use only the approved evidence and urgency rules documented in this specification.

No mathematical fusion formula is approved or implemented for v1.

### Evidence and Recommendation Semantics

#### `evidence_status`

`evidence_status` explicitly communicates whether sufficient approved evidence exists for the recommendation.

It must distinguish between:

- approved urgency-relevant or severity-relevant evidence being available
- no approved urgency-relevant evidence being available

When no approved urgency-relevant evidence is available:

- urgency remains undefined
- insufficient evidence must be explicitly represented
- the system must not silently assign Routine

This field does not introduce new thresholds or clinical rules.

#### `evidence`

`evidence` records observable or supporting evidence already produced by the existing model/output pipeline.

It may preserve information needed to explain the evidence status or recommendation.

This field must not:

- generate new disease-specific clinical evidence
- infer an unapproved severity mapping
- infer an unapproved urgency mapping
- introduce a fusion mechanism

#### `recommendation`

`recommendation` communicates the result of the available approved evidence.

When insufficient approved evidence exists for an urgency determination, the recommendation must explicitly communicate insufficient evidence / urgency undefined rather than silently assigning Routine.

This field must not independently introduce:

- Dog Eye -> severity mapping
- Dog Eye -> urgency mapping
- confidence -> urgency conversion
- new Soon triggers
- new Urgent triggers
- additional Emergency triggers
- mathematical fusion
- unsupported final clinical decisions

## Final Recommendation Schema

The Fusion / Urgency output schema is approved at the field level.

The schema may contain:

- `species`
- `body_area`
- `condition`
- `confidence`
- `confidence_level`
- `uncertain`
- `severity`
- `urgency`
- `evidence_status`
- `evidence`
- `recommendation`

### Field constraints

- `species` identifies the species being evaluated.
- `body_area` identifies the relevant body area.
- `condition` records the identified condition/evidence.
- `confidence` may contain numeric model confidence.
- `confidence_level` may contain a categorical confidence representation if required by the project. Its categories and thresholds are not defined by this specification.
- `uncertain` explicitly represents the existing uncertainty state.
- `severity` may be present but must remain undefined/unpopulated when no approved severity rule or evidence exists.
- `urgency` may be present but must remain undefined when no approved urgency evidence or rule exists.
- `evidence_status` explicitly represents evidence states such as insufficient evidence.
- `evidence` may preserve the supporting evidence behind the recommendation.
- `recommendation` contains the user-facing recommendation without introducing new urgency or fusion rules.

The presence of a field does not imply that a value must always be populated.

This schema definition does not establish:

- a fusion formula
- confidence thresholds
- disease-specific urgency mappings
- Dog Eye severity/urgency mappings
- additional urgency assignment rules

## Versioning

The Fusion / Urgency specification must be finalized before implementation.

The Dog Eye model and its inference contract remain frozen independently of this specification.

Any future Fusion / Urgency implementation must conform to the approved specification without modifying the frozen Dog Eye inference contract.
