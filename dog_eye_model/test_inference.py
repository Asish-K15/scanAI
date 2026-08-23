from pathlib import Path

import math

from inference import DogEyeModel


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_MANAGER = PROJECT_DIR.parent / "dataset_manager"


BORDERLINE_IMAGE = (
    DATASET_MANAGER
    / "expansion"
    / "dog_conjunctivitis"
    / "test_D0_0e49d71d-60a5-11ec-8402-0a7404972c70_jpg.rf.WV8mVlnd5VD3jq4BEaiG.jpg"
)

STRONG_IMAGE = (
    DATASET_MANAGER
    / "expansion"
    / "dog_entropion"
    / "test_D31_0e5e1ccd-60a5-11ec-8402-0a7404972c70_jpg.rf.r80vfrNSuy5KeEZAUJbX.jpg"
)


def validate_result(result):

    required_keys = {
        "condition",
        "confidence",
        "confidence_level",
        "uncertain",
        "probabilities",
        "model",
        "model_version",
        "engine",
        "screening_only",
    }

    assert required_keys.issubset(result.keys())

    assert result["condition"] in {
        "conjunctivitis",
        "entropion",
    }

    assert 0.0 <= result["confidence"] <= 1.0

    probabilities = result["probabilities"]

    assert "conjunctivitis" in probabilities
    assert "entropion" in probabilities

    probability_sum = (
        probabilities["conjunctivitis"]
        + probabilities["entropion"]
    )

    assert math.isclose(
        probability_sum,
        1.0,
        rel_tol=1e-5,
        abs_tol=1e-5,
    )

    assert result["confidence_level"] in {
        "low",
        "moderate",
        "high",
    }

    assert isinstance(
        result["uncertain"],
        bool
    )

    assert result["screening_only"] is True


def main():

    print("=" * 60)
    print("SCANAI DOG EYE INFERENCE TEST")
    print("=" * 60)

    assert BORDERLINE_IMAGE.exists()
    assert STRONG_IMAGE.exists()

    model = DogEyeModel()

    print()
    print("Testing borderline image...")

    borderline = model.predict(
        BORDERLINE_IMAGE
    )

    validate_result(borderline)

    print(borderline)

    assert borderline["confidence_level"] == "low"
    assert borderline["uncertain"] is True

    print()
    print("Borderline test: PASS")

    print()
    print("Testing strong image...")

    strong = model.predict(
        STRONG_IMAGE
    )

    validate_result(strong)

    print(strong)

    assert strong["condition"] == "entropion"
    assert strong["confidence_level"] == "high"
    assert strong["uncertain"] is False

    print()
    print("Strong prediction test: PASS")

    print()
    print("=" * 60)
    print("ALL INFERENCE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()