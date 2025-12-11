import importlib
import pkgutil


def discover_modules(package) -> dict:
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

    return package.REGISTRY
