from types import ModuleType

import importlib
import pkgutil

from Model import Model


def discover_modules(package: str | ModuleType) -> dict[str, type[Model]]:
    """
    Imports all modules in the given package.

    Args:
        package (module): The package to scan.
    """
    if isinstance(package, str):
        package = importlib.import_module(package)

    # Testing if is a package or just a module.
    if not hasattr(package, "__path__"):
        return {}

    for _, name, _ in pkgutil.iter_modules(package.__path__):
        full_name = f"{package.__name__}.{name}"
        importlib.import_module(full_name)
    assert isinstance(package.REGISTRY, dict)
    return package.REGISTRY
