import pkgutil
import os


# All newly implemented Fusion methods should have a 'device' and a 'model' attribute, and a 'forward' method that
# takes two images as input and returns the fused image, the input visible image and the input infrared image.


__all__ = []
__methods__ = {}

# Iterate through all modules in this package
for module_info in pkgutil.iter_modules([os.path.dirname(__file__)]):
    name = module_info.name
    __all__.append(name.lower())
    __methods__[name.lower()] = f"{__name__}.{name}".replace('fusion_framework.methods', '')
    # __import__(f"{__name__}.{name}")
