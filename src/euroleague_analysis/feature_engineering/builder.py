"""Feature engineering builder module."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .database import FeatureDatabase
from .views.stats import (
    GameContextView,
    PlayByPlayView,
    PlayerTiersView,
    RollingStatsView,
    ShotPatternsView,
)

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Handles the feature engineering pipeline"""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.db = FeatureDatabase(config.db_path)
        self._setup_views()

    def _setup_views(self) -> None:
        """Initialize SQL views"""
        self.views = [
            RollingStatsView(),
            ShotPatternsView(),
            PlayByPlayView(),
        ]
        logger.info(f"Initialized {len(self.views)} feature views")

    def create_feature_views(self) -> None:
        """Create all feature views for each season"""
        logger.info("Creating feature views...")
        self.db.connect()

        try:
            for season in self.config.seasons:
                logger.info(f"Processing season {season}")
                format_args = {
                    "data_dir": self.config.data_dir,
                    "season": season,
                }

                for view in self.views:
                    view_name = f"{view.name}_{season}"
                    logger.info(f"Creating view: {view_name}")
                    view.create(self.db, **format_args)
                    logger.info(f"Successfully created view: {view_name}")

            logger.info("All feature views created successfully")

        except Exception as e:
            logger.error(f"Error creating views: {e}")
            raise

    def create_final_features(self) -> pd.DataFrame:
        """Generate final feature set combining all seasons"""
        if not self.db.conn:
            raise RuntimeError("Database connection not initialized")

        logger.info("Creating final features...")
        try:
            # Execute final features query
            df = self.db.query_to_df(self._get_final_features_query())

            # Validate features
            self._validate_features(df)

            # Save to parquet
            output_path = self.config.output_dir / "features.parquet"
            df.to_parquet(str(output_path))
            logger.info(f"Saved features to {output_path}")

            return df

        except Exception as e:
            logger.error(f"Error creating final features: {e}")
            raise

    def save_metadata(self) -> None:
        """Save feature metadata"""
        metadata = {
            "features_file": str(self.config.output_dir / "features.parquet"),
            "database_file": self.config.db_path,
            "num_features": len(self._get_feature_columns()),
            "num_samples": self._get_num_samples(),
            "seasons": self.config.seasons,
        }

        with open("feature_outputs.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved feature metadata")

    def close(self) -> None:
        """Close database connection"""
        self.db.close()
        logger.info("Closed database connection")

    def _get_final_features_query(self) -> str:
        """Get the SQL query for final feature set combining all seasons"""
        season_queries = []

        for season in self.config.seasons:
            season_query = f"""
            SELECT
                r.Season,
                r.Phase,
                r.Round,
                r.Gamecode,
                r.Player_ID,
                -- Basic features
                r.minutes_played,
                r.is_starter,
                r.is_home,
                r.Points,
                r.PIR,
                r.fg_percentage,
                r.ft_percentage,
                r.ast_to_turnover,
                
                -- Rolling averages
                r.pir_ma3,
                r.points_ma3,
                r.minutes_ma3,
                
                -- Shot patterns
                s.fg_percentage as shot_fg_percentage,
                s.fg_percentage_2pt,
                s.fg_percentage_3pt,
                s.three_point_rate,
                s.fastbreak_rate,
                s.second_chance_rate,
                
                -- Game flow features from playbyplay
                p.clutch_plays,
                p.clutch_scores,
                p.first_quarter_plays,
                p.fourth_quarter_plays,
                p.close_game_plays,
                p.unique_play_types,
                p.consecutive_positive_plays,
                
                -- Usage patterns from playbyplay
                p.assist_rate,
                p.shot_attempt_rate,
                p.defensive_play_rate,
                p.turnover_rate,
                
                -- Form and consistency
                r.improving_form,
                r.pir_std3,
                r.pir_vs_season_avg,
                r.pir_rank_in_game
                
            FROM player_stats_features_{season} r
            LEFT JOIN shot_patterns_{season} s
                ON r.Player_ID = s.player_id
                AND r.Gamecode = s.gamecode
            LEFT JOIN playbyplay_features_{season} p
                ON r.Player_ID = p.PLAYER_ID
                AND r.Gamecode = p.Gamecode
            WHERE r.minutes_played > 0
            """
            season_queries.append(season_query)

        combined_query = " UNION ALL ".join(season_queries)
        return f"""
        WITH all_seasons AS (
            {combined_query}
        )
        SELECT * FROM all_seasons
        ORDER BY Season, Round, Gamecode, PIR DESC
        """

    def _validate_features(self, df: pd.DataFrame) -> None:
        """Validate the generated features"""
        logger.info("Validating features...")

        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in features")

        # Check for infinite values
        infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if infinite_values > 0:
            logger.warning(f"Found {infinite_values} infinite values in features")

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows in features")

        logger.info("Feature validation completed")

    def _get_feature_columns(self) -> List[str]:
        """Get list of feature columns"""
        return [
            # Basic features
            "minutes_played",
            "is_starter",
            "is_home",
            "Points",
            "PIR",
            "fg_percentage",
            "ft_percentage",
            "ast_to_turnover",
            # Rolling averages
            "pir_ma3",
            "points_ma3",
            "minutes_ma3",
            # Shot patterns
            "shot_fg_percentage",
            "fg_percentage_2pt",
            "fg_percentage_3pt",
            "three_point_rate",
            "fastbreak_rate",
            "second_chance_rate",
            # Game flow features
            "clutch_plays",
            "clutch_scores",
            "first_quarter_plays",
            "fourth_quarter_plays",
            "close_game_plays",
            "unique_play_types",
            "consecutive_positive_plays",
            # Usage patterns
            "assist_rate",
            "shot_attempt_rate",
            "defensive_play_rate",
            "turnover_rate",
            # Form and consistency
            "improving_form",
            "pir_std3",
            "pir_vs_season_avg",
            "pir_rank_in_game",
        ]

    def _get_num_samples(self) -> int:
        """Get number of samples in the dataset"""
        try:
            total = 0
            for season in self.config.seasons:
                count = self.db.conn.execute(
                    f"SELECT COUNT(*) FROM player_stats_features_{season}"
                ).fetchone()[0]
                total += count
            return total
        except Exception as e:
            logger.error(f"Error getting sample count: {e}")
            return 0
