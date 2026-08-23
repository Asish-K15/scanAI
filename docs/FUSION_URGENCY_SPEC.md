# ScanAI Fusion / Urgency Specification

## Status

Specification draft — implementation blocked until undefined rules are explicitly approved.

## Purpose

The Fusion / Urgency layer converts confidence-tagged vision-model outputs into application-level triage recommendations.

It is separate from the vision inference models and must remain independently versionable.

## Design Principles

1. Vision models provide probabilistic predictions.
2. Every vision prediction is treated as estimated from a photo.
3. Model confidence must not automatically be interpreted as disease severity.
4. Recommendation logic uses deterministic rule-based clinical logic together with model outputs.
5. ScanAI is a triage / first-look tool and is not a replacement for veterinary diagnosis.
6. Vision inference and recommendation logic remain separate components.

## Currently Defined Severity

Wound severity is defined as:

- mild
- moderate
- severe

This represents severity grading rather than exact wound-type diagnosis.

## Documented Urgency Examples

A severe deep-tissue laceration with active hemorrhage is an example of high urgency.

A low-grade chronic dermatitis or minor superficial abrasion may result in routine monitoring / non-urgent consultation.

These examples establish the intended rule-based approach but do not define a complete numerical urgency system.

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

## Undefined — Requires Explicit Specification

The following are not currently defined:

- exact urgency levels
- exact urgency thresholds
- confidence thresholds for urgency
- mathematical fusion formula
- Dog Eye severity mapping
- Dog Eye urgency mapping
- rules combining confidence, condition, and severity

These must not be invented or inferred from the Dog Eye model.

## Dog Eye Confidence Thresholds

The Dog Eye model currently exposes:

- high: confidence >= 0.80
- moderate: confidence >= 0.60
- low: confidence < 0.60

These are engineering confidence levels only.

They are NOT clinical severity or urgency thresholds.

## Output

The final recommendation schema has not yet been finalized.

Potential recommendation concepts include:

- severity
- urgency
- first-aid guidance
- veterinary referral prompt

Exact fields and rules require explicit specification.

## Versioning

The Fusion / Urgency specification must be finalized before implementation.

The Dog Eye model and its inference contract remain frozen independently of this specification.
