"""Feature engineering builder module."""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import List

from .config import FeatureConfig
from .database import FeatureDatabase
from .views.stats import (
    RollingStatsView,
    ShotPatternsView,
    GameContextView,
    PlayByPlayView,
    PlayerTiersView
)

logger = logging.getLogger(__name__)

class FeatureBuilder:
    """Handles the feature engineering pipeline"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.db = FeatureDatabase(config.db_path)
        self._setup_views()
        
    def _setup_views(self):
        """Initialize SQL views"""
        self.views = [
            RollingStatsView(),
            ShotPatternsView(),
            GameContextView(),
            PlayByPlayView(),
            PlayerTiersView()
        ]
        
    def create_feature_views(self):
        """Create all feature views"""
        self.db.connect()
        
        try:
            format_args = {
                "data_dir": self.config.data_dir,
                "season": self.config.season
            }
            
            for view in self.views:
                view.create(self.db, **format_args)
                
        except Exception as e:
            logger.error(f"Error creating views: {e}")
            raise
            
    def create_final_features(self) -> pd.DataFrame:
        """Generate final feature set"""
        if not self.db.conn:
            raise RuntimeError("Database connection not initialized")
            
        try:
            # Execute final features query
            df = self.db.query_to_df(self._get_final_features_query())
            
            # Save to parquet
            output_path = self.config.output_dir / "features.parquet"
            df.to_parquet(str(output_path))
            logger.info(f"Saved features to {output_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating final features: {e}")
            raise
            
    def save_metadata(self):
        """Save feature metadata"""
        metadata = {
            "features_file": str(self.config.output_dir / "features.parquet"),
            "database_file": self.config.db_path,
        }
        
        with open("feature_outputs.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("Saved feature metadata")
        
    def close(self):
        """Close database connection"""
        self.db.close()
        
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
