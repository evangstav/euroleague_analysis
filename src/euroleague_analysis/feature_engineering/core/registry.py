"""Feature transformer registry."""

from typing import Dict, List, Type

from .base import FeatureTransformer


class FeatureRegistry:
    """Central registry for feature transformers."""

    def __init__(self):
        self._transformers: Dict[str, Type[FeatureTransformer]] = {}

    def register(self, name: str, transformer_cls: Type[FeatureTransformer]) -> None:
        """Register a new transformer class.

        Args:
            name: Unique name for the transformer
            transformer_cls: Transformer class to register
        """
        if name in self._transformers:
            raise ValueError(f"Transformer {name} already registered")
        self._transformers[name] = transformer_cls

    def get_transformer(self, name: str, **kwargs) -> FeatureTransformer:
        """Get instance of registered transformer.

        Args:
            name: Name of transformer to get
            **kwargs: Arguments to pass to transformer constructor

        Returns:
            New instance of requested transformer
        """
        if name not in self._transformers:
            raise KeyError(f"No transformer registered with name: {name}")
        return self._transformers[name](**kwargs)

    def list_transformers(self) -> List[str]:
        """Get list of registered transformer names."""
        return list(self._transformers.keys())


# Global registry instance
registry = FeatureRegistry()
