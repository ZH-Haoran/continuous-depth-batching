"""Model-specific adapters for continuous depth batching."""

from .huginn import HuginnCDBAdapter
from .ouro import OuroCDBAdapter

__all__ = ["HuginnCDBAdapter", "OuroCDBAdapter"]
