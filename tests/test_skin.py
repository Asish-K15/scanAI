import unittest

from app.services.skin import (
    SkinModelNotAvailableError,
    get_skin_model,
)


class TestSkinModelInterface(unittest.TestCase):

    def test_skin_model_is_not_faked_before_integration(self):
        with self.assertRaises(SkinModelNotAvailableError):
            get_skin_model()


if __name__ == "__main__":
    unittest.main()