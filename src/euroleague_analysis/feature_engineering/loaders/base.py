"""Base data loader implementation."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    def __init__(self, data_dir: Path):
        """Initialize loader with data directory.

        Args:
            data_dir: Path to directory containing data files
        """
        self.data_dir = data_dir

    @abstractmethod
    def load_data(self, season: str, data_type: str) -> pl.DataFrame:
        """Load data for a specific season and type.

        Args:
            season: Season identifier (e.g., "2023")
            data_type: Type of data to load (e.g., "player_stats")

        Returns:
            Polars DataFrame with loaded data
        """
        pass


class ParquetLoader(DataLoader):
    """Loader implementation for Parquet files."""

    def __init__(self, data_dir: Path):
        """Initialize Parquet loader.

        Args:
            data_dir: Path to directory containing Parquet files
        """
        super().__init__(data_dir)
        self.file_patterns = {
            "player_stats": "player_stats_{season}.parquet",
            "shot_data": "shot_data_{season}.parquet",
            "playbyplay": "playbyplay_data_{season}.parquet",
        }

    def load_data(self, season: str, data_type: str) -> pl.DataFrame:
        """Load data from Parquet file.

        Args:
            season: Season identifier (e.g., "2023")
            data_type: Type of data to load

        Returns:
            Polars DataFrame with loaded data

        Raises:
            FileNotFoundError: If data file is not found
            ValueError: If data type is not supported
        """
        if data_type not in self.file_patterns:
            raise ValueError(f"Unsupported data type: {data_type}")

        file_pattern = self.file_patterns[data_type]
        file_path = self.data_dir / file_pattern.format(season=season)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Expected file pattern: {file_pattern}"
            )

        try:
            logger.info(f"Loading {data_type} data from {file_path}")
            df = pl.read_parquet(file_path)
            logger.info(f"Successfully loaded {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error loading {data_type} data: {e}")
            raise


class CachedLoader(DataLoader):
    """Loader with in-memory caching."""

    def __init__(self, loader: DataLoader):
        """Initialize cached loader.

        Args:
            loader: Underlying loader to use for actual loading
        """
        self.loader = loader
        self._cache: Dict[str, pl.DataFrame] = {}

    def load_data(self, season: str, data_type: str) -> pl.DataFrame:
        """Load data with caching.

        Args:
            season: Season identifier
            data_type: Type of data to load

        Returns:
            Polars DataFrame with loaded data
        """
        cache_key = f"{season}_{data_type}"

        if cache_key not in self._cache:
            self._cache[cache_key] = self.loader.load_data(season, data_type)

        return self._cache[cache_key]


class SeasonLoader(DataLoader):
    """Loader that handles season-specific data requirements."""

    def __init__(self, data_dir: Path, use_cache: bool = True):
        """Initialize season loader.

        Args:
            data_dir: Path to data directory
            use_cache: Whether to use caching
        """
        super().__init__(data_dir)
        self.loader = create_loader(data_dir, use_cache=use_cache)
        self.required_data_types = ["player_stats", "shot_data", "playbyplay"]

    def load_data(self, season: str, data_type: str) -> pl.DataFrame:
        """Load data for a specific season and type.

        Args:
            season: Season identifier
            data_type: Type of data to load

        Returns:
            Polars DataFrame with loaded data

        Raises:
            ValueError: If data type is not supported
            FileNotFoundError: If data file is not found
        """
        if data_type not in self.required_data_types:
            raise ValueError(f"Unsupported data type: {data_type}")

        return self.loader.load_data(season, data_type)

    def load_season_data(self, season: str) -> Dict[str, pl.DataFrame]:
        """Load all required data for a season.

        Args:
            season: Season identifier

        Returns:
            Dictionary mapping data types to DataFrames

        Raises:
            FileNotFoundError: If any required data file is missing
        """
        data = {}

        for data_type in self.required_data_types:
            try:
                data[data_type] = self.load_data(season, data_type)
                logger.info(f"Loaded {len(data[data_type])} rows of {data_type} data")
            except FileNotFoundError as e:
                logger.error(f"Missing data file for {data_type}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error loading {data_type} data: {e}")
                raise

        return data


def create_loader(data_dir: Path, use_cache: bool = True) -> DataLoader:
    """Create appropriate loader instance.

    Args:
        data_dir: Path to data directory
        use_cache: Whether to use caching

    Returns:
        Configured data loader instance
    """
    loader = ParquetLoader(data_dir)
    return CachedLoader(loader) if use_cache else loader
