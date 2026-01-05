from enum import StrEnum


class VisionTag(StrEnum):
    VECTOR_BASED = "vector_based"
    IMAGE_BASED = "image_based"
    NO_RECONSTRUCTION = "no_reconstruction"


class MemoryTag(StrEnum):
    pass


class ControllerTag(StrEnum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    STOCHASTIC = "stochastic"
