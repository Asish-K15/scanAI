from dog_eye_model.inference import DogEyeModel

_model = None


def get_dog_eye_model():
    global _model

    if _model is None:
        _model = DogEyeModel()

    return _model
