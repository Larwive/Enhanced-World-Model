from Model import Model

REGISTRY = {}

class VisionModel(Model):
    """
    The base class for vision (V) models.
    """

    def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            REGISTRY[cls.__name__] = cls
