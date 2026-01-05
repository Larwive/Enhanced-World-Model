from src.utils.registry import discover_modules

CONTROLLER_REGISTRY = discover_modules("controller")


def test_controller_creation(controller_input_args):
    for name, cls in CONTROLLER_REGISTRY.items():
        controller = cls(**controller_input_args)
        assert controller is not None
        assert name == controller.__class__.__name__
        # TODO: Uncomment the following line after #62 is solved.
        # assert isinstance(controller.tags, frozenset), f"Tags must be defined for {name}, even if empty."
