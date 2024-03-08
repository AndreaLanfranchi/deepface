from typing import Any, Dict

import importlib
import inspect
import pkgutil

from deepface.commons.logger import Logger

Logger = Logger()


def get_derived_classes(package: Any, base_class: type) -> Dict[str, type]:
    """
    Get classes inheriting from base_class in module.
    """

    results: Dict[str, type] = {}
    for _, module_name, _ in pkgutil.walk_packages(package.__path__):
        module = importlib.import_module(name=f"{package.__name__}.{module_name}")

        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                if issubclass(obj, base_class) and obj is not base_class:
                    Logger.debug(
                        f"Found class [{obj.__name__}] in module [{module.__name__}]"
                    )
                    key_value: str = str(
                        module.__name__.rsplit(".", maxsplit=1)[-1]
                    ).lower()
                    results[key_value] = obj
                    break  # Only one detector per module

    return results
