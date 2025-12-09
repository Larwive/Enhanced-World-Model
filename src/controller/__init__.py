from Model import Model

REGISTRY = {}

class ControllerModel(Model):
    """
    The base class for controller (C) models.
    """

    def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            REGISTRY[cls.__name__] = cls
