import pkgutil
import os

__all__ = []
__methods__ = {}

# Iterate through all modules in this package
for module_info in pkgutil.iter_modules([os.path.dirname(__file__)]):
    name = module_info.name
    __all__.append(name.lower())
    __methods__[name.lower()] = f"{__name__}.{name}".replace('methods', '')
    # __import__(f"{__name__}.{name}")
