class SkinModelNotAvailableError(RuntimeError):
    """Raised when the Skin model has not yet been integrated."""


def get_skin_model():
    """
    Return the integrated Skin model.

    The actual trained Skin model will be supplied by Partner B
    after model evaluation and approval.

    No predictions are generated here until that model is integrated.
    """
    raise SkinModelNotAvailableError(
        "Skin model is not integrated yet."
    )