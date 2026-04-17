import importlib.metadata as metadata

try:
    __version__ = metadata.version(__package__ or 'prism_imageproc')
except metadata.PackageNotFoundError:
    # Fallback version if package metadata is not found. 
    # This should be updated if the version is changed in pyproject.toml.
    __version__ = '0.0.2'

from .straighten_image import (
    ImageStraightener,
    MappedImage,
)
from . import utils
from .internals import PaddingMode

__all__ = [
    "ImageStraightener",
    "MappedImage",
    "internals",
    "utils",
    "__version__",
]