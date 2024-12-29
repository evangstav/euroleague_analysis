"""Feature engineering builder module."""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from .config import FeatureConfig
from .database import FeatureDatabase
from .views.stats import (
    RollingStatsView,
    ShotPatternsView,
    GameContextView,
    PlayByPlayView,
    PlayerTiersView,
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
            GameContextView(),
            PlayByPlayView(),
            PlayerTiersView(),
        ]
        logger.info(f"Initialized {len(self.views)} feature views")

    def create_feature_views(self) -> None:
        """Create all feature views"""
        logger.info("Creating feature views...")
        self.db.connect()

        try:
            format_args = {
                "data_dir": self.config.data_dir,
                "season": self.config.season,
            }

            for view in self.views:
                logger.info(f"Creating view: {view.name}")
                view.create(self.db, **format_args)
                logger.info(f"Successfully created view: {view.name}")

            logger.info("All feature views created successfully")

        except Exception as e:
            logger.error(f"Error creating views: {e}")
            raise

    def create_final_features(self) -> pd.DataFrame:
        """Generate final feature set"""
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
        }

        with open("feature_outputs.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved feature metadata")

    def close(self) -> None:
        """Close database connection"""
        self.db.close()
        logger.info("Closed database connection")

    def _get_final_features_query(self) -> str:
        """Get the SQL query for final feature set"""
        return """
        SELECT 
            r.Season,
            r.Phase,
            r.Round,
            r.Gamecode,
            r.Player_ID,
            -- ... rest of the query ...
        FROM player_stats_features r
        LEFT JOIN shot_patterns s 
            ON r.Player_ID = s.Player_ID 
            AND r.Gamecode = s.Gamecode
        LEFT JOIN playbyplay_features p 
            ON r.Player_ID = p.PLAYER_ID 
            AND r.Gamecode = p.Gamecode
        LEFT JOIN game_context g 
            ON r.Gamecode = g.Gamecode
            AND r.Season = g.Season
        LEFT JOIN player_tiers_view h
            ON r.Player_ID = h.Player_ID
            AND r.Gamecode = h.Gamecode
        WHERE r.minutes_played > 0
        ORDER BY r.Season, r.Round, r.Gamecode, r.PIR DESC
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
        # This should return the actual feature columns used in the model
        return [
            "PIR", "minutes_played", "Points", "fg_percentage", 
            "ft_percentage", "ast_to_turnover", "pir_ma3", 
            "points_ma3", "minutes_ma3", "is_starter", "is_home"
        ]

    def _get_num_samples(self) -> int:
        """Get number of samples in the dataset"""
        try:
            with self.db.conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM player_stats_features")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting sample count: {e}")
            return 0
