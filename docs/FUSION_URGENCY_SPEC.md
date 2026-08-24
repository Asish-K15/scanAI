# ScanAI Fusion / Urgency Specification

## Status

Specification consolidated from reviewed and approved project decisions.

Implementation remains blocked until the remaining undefined rules are explicitly specified and approved.

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

Urgent is an approved application-level category.

No specific observable condition is currently documented or approved as an Urgent trigger.

Therefore:

- Urgent assignment criteria remain undefined.
- No unsupported clinical condition should be mapped to Urgent.

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
- If no urgency-relevant evidence is available, the final behavior remains undefined until explicitly approved.

Important distinctions:

- uncertainty ≠ low severity
- uncertainty ≠ high urgency

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

## Explicitly Undefined

The following remain undefined and must not be invented:

- Soon assignment criteria
- Urgent assignment criteria
- Additional Emergency evidence
- Final behavior when no urgency-relevant evidence exists
- Final recommendation schema
- Mathematical fusion formula
- Dog Eye severity mapping
- Dog Eye urgency mapping
- Rules combining confidence, condition, and severity
- General conflict-resolution formula

## Prohibited Assumptions

The following must not be introduced without explicit project approval:

- confidence → urgency conversion
- Dog Eye confidence threshold → clinical severity
- Dog Eye confidence threshold → clinical urgency
- disease-specific Dog Eye urgency mappings
- unsupported disease-specific severity mappings
- mathematical fusion formulas presented as previously agreed requirements
- unsupported clinical rules presented as existing ScanAI requirements

## Output

Potential recommendation concepts include:

- severity
- urgency
- first-aid guidance
- veterinary referral prompt

The exact final recommendation schema has not yet been finalized.

## Versioning

The Fusion / Urgency specification must be finalized before implementation.

The Dog Eye model and its inference contract remain frozen independently of this specification.

Any future Fusion / Urgency implementation must conform to the approved specification without modifying the frozen Dog Eye inference contract.
