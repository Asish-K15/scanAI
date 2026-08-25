import unittest

import numpy as np

from dog_eye_model.inference import DogEyeModel


class TestDogEyeModel(unittest.TestCase):

    def test_softmax_probabilities_sum_to_one(self):
        logits = np.array(
            [[2.0, 1.0]],
            dtype=np.float32,
        )

        probabilities = DogEyeModel.softmax(logits)[0]

        self.assertAlmostEqual(
            float(np.sum(probabilities)),
            1.0,
            places=6,
        )

        self.assertGreater(probabilities[0], probabilities[1])


    def test_confidence_level_thresholds(self):
        self.assertEqual(
            DogEyeModel.get_confidence_level(0.90),
            "high",
        )
        self.assertEqual(
            DogEyeModel.get_confidence_level(0.80),
            "high",
        )
        self.assertEqual(
            DogEyeModel.get_confidence_level(0.79),
            "moderate",
        )
        self.assertEqual(
            DogEyeModel.get_confidence_level(0.60),
            "moderate",
        )
        self.assertEqual(
            DogEyeModel.get_confidence_level(0.59),
            "low",
        )


    def test_softmax_preserves_argmax(self):
        logits = np.array(
            [[-1.0, 3.0]],
            dtype=np.float32,
        )

        probabilities = DogEyeModel.softmax(logits)[0]

        self.assertEqual(
            int(np.argmax(logits[0])),
            int(np.argmax(probabilities)),
        )


    def test_softmax_handles_large_logits(self):
        logits = np.array(
            [[1000.0, 999.0]],
            dtype=np.float32,
        )

        probabilities = DogEyeModel.softmax(logits)[0]

        self.assertTrue(
            np.all(np.isfinite(probabilities))
        )

        self.assertAlmostEqual(
            float(np.sum(probabilities)),
            1.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
