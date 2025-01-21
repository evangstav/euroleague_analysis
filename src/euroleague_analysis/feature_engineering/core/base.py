"""Base classes for feature engineering."""

from abc import ABC, abstractmethod
from typing import List, Optional

import polars as pl


class FeatureTransformer(ABC):
    """Base class for all feature transformers."""

    def __init__(self, name: str, required_columns: Optional[List[str]] = None):
        """Initialize transformer.

        Args:
            name: Unique identifier for this transformer
            required_columns: List of column names required by this transformer
        """
        self.name = name
        self.required_columns = required_columns or []

    def validate_columns(self, df: pl.DataFrame) -> None:
        """Validate required columns exist in DataFrame."""
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @abstractmethod
    def fit(self, df: pl.DataFrame) -> "FeatureTransformer":
        """Fit transformer to data (if needed).

        Args:
            df: Input DataFrame

        Returns:
            self
        """
        return self

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform input data.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame with new features
        """
        pass

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this transformer."""
        return []


class FeatureCollection(FeatureTransformer):
    """A collection of feature transformers to be applied sequentially."""

    def __init__(self, name: str, transformers: List[FeatureTransformer]):
        """Initialize collection.

        Args:
            name: Collection name
            transformers: List of transformers to apply
        """
        super().__init__(name)
        self.transformers = transformers

    def fit(self, df: pl.DataFrame) -> "FeatureCollection":
        """Fit all transformers in sequence."""
        for transformer in self.transformers:
            transformer.fit(df)
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all transformers in sequence."""
        result = df
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result

    def get_feature_names(self) -> List[str]:
        """Get combined list of feature names from all transformers."""
        names = []
        for transformer in self.transformers:
            names.extend(transformer.get_feature_names())
        return names
