"""Feature engineering pipeline."""

import logging
from pathlib import Path
from typing import List, Optional

import polars as pl

from .core.base import FeatureTransformer
from .core.registry import registry
from .features.combined_stats import CombinedStatsTransformer
from .features.matchups import PlayerMatchupTransformer
from .features.playbyplay import PlayByPlayTransformer
from .features.shot_patterns import ShotPatternsTransformer
from .loaders.base import SeasonLoader

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrate the feature engineering process."""

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        seasons: List[str],
        transformers: Optional[List[str]] = None,
    ):
        """Initialize pipeline.

        Args:
            data_dir: Directory containing raw data
            output_dir: Directory for output files
            seasons: List of seasons to process
            transformers: List of transformer names to use (default: all)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seasons = seasons
        self.loader = SeasonLoader(data_dir)

        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Please ensure the data_dir path in params.yaml points to your data directory."
            )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize transformers
        self.transformers = self._setup_transformers(transformers)
        logger.info(f"Initialized pipeline with {len(self.transformers)} transformers")

    def _setup_transformers(
        self, transformer_names: Optional[List[str]] = None
    ) -> List[FeatureTransformer]:
        """Set up feature transformers.

        Args:
            transformer_names: List of transformer names to use

        Returns:
            List of transformer instances
        """
        # Default transformer sequence if none specified
        if transformer_names is None:
            transformer_names = [
                "combined_stats",
                "shot_patterns",
                "playbyplay",
                "position_matchups",
            ]

        transformers = []
        for name in transformer_names:
            try:
                transformer = registry.get_transformer(name)
                transformers.append(transformer)
                logger.info(f"Added transformer: {name}")
            except KeyError:
                logger.warning(f"Transformer not found: {name}")

        return transformers

    def _load_season_data(self, season: str) -> dict:
        """Load all required data for a season.

        Args:
            season: Season identifier

        Returns:
            Dictionary of DataFrames for different data types
        """
        try:
            return self.loader.load_season_data(season)
        except Exception as e:
            logger.error(f"Error loading season {season} data: {e}")
            raise

    def process_season(self, season: str) -> pl.DataFrame:
        """Process a single season.

        Args:
            season: Season identifier

        Returns:
            DataFrame with computed features
        """
        logger.info(f"Processing season {season}")

        try:
            # Load all required data
            data = self._load_season_data(season)

            # Apply each transformer to its specific input data
            transformed_data = {}
            for transformer in self.transformers:
                logger.info(f"Applying transformer: {transformer.name}")

                # Select appropriate input data for transformer
                if transformer.name == "shot_patterns":
                    input_df = data["shot_data"].rename({"ID_PLAYER": "Player_ID"})
                elif transformer.name in ["playbyplay", "position_matchups"]:
                    input_df = data["playbyplay"].rename({"PLAYER_ID": "Player_ID"})
                else:
                    input_df = data["player_stats"]

                # Apply transformation
                try:
                    transformed_df = transformer.transform(input_df)
                    transformed_data[transformer.name] = transformed_df
                except Exception as e:
                    logger.error(f"Error in transformer {transformer.name}: {e}")
                    raise

            # Start with base player stats and join all transformed results
            features_df = data["player_stats"]
            for transformer_name, transformed_df in transformed_data.items():
                logger.info(f"Joining {transformer_name} features")
                join_keys = ["Player_ID", "Gamecode"]
                features_df = features_df.join(transformed_df, on=join_keys, how="left")
            # Fill any nulls from joins
            features_df = features_df.fill_null(0)
            return features_df

        except Exception as e:
            logger.error(f"Error processing season {season}: {e}")
            raise

    def run(self) -> pl.DataFrame:
        """Run the full feature engineering pipeline.

        Returns:
            DataFrame with all computed features
        """
        logger.info("Starting feature engineering pipeline")

        # Process each season
        season_dfs = []
        for season in self.seasons:
            try:
                df = self.process_season(season)
                season_dfs.append(df)
            except Exception as e:
                logger.error(f"Error processing season {season}: {e}")
                raise

        # Combine all seasons
        logger.info("Combining seasons")
        result = pl.concat(season_dfs)

        # Save to parquet
        output_path = self.output_dir / "features.parquet"
        result.write_parquet(str(output_path))
        logger.info(f"Saved features to {output_path}")

        # Log summary
        logger.info(
            f"Generated {len(result.columns)} features for {len(result)} samples"
        )

        return result
