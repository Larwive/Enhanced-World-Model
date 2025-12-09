from Model import Model

REGISTRY = {}

class MemoryModel(Model):
    """
    The base class for memory (M) models.
    """

    def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            REGISTRY[cls.__name__] = cls
