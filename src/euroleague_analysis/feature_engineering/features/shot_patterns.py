"""Shot pattern features."""

from typing import List

import numpy as np
import polars as pl

from ..core.base import FeatureTransformer
from ..core.registry import registry


class ShotPatternsTransformer(FeatureTransformer):
    """Analyze shot patterns and tendencies."""

    def __init__(self):
        super().__init__(
            name="shot_patterns",
            required_columns=[
                "Player_ID",
                "Gamecode",
                "FASTBREAK",
                "SECOND_CHANCE",
                "COORD_X",
                "COORD_Y",
                "ZONE",
                "ID_ACTION",
                "ACTION",
            ],
        )

    def fit(self, df: pl.DataFrame) -> "ShotPatternsTransformer":
        """No fitting required for shot patterns."""
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate shot pattern features.

        Args:
            df: Input DataFrame with shot data

        Returns:
            DataFrame with shot pattern features
        """
        self.validate_columns(df)

        # Cast string columns to integers
        df = df.with_columns(
            [
                pl.col("FASTBREAK").cast(pl.Int64).alias("FASTBREAK"),
                pl.col("SECOND_CHANCE").cast(pl.Int64).alias("SECOND_CHANCE"),
            ]
        )

        # Add basic shot metrics
        df = df.with_columns(
            [
                # Shot outcomes based on ID_ACTION and ACTION
                (
                    pl.col("ACTION").str.contains("Two Pointer|Three Pointer")
                    & ~pl.col("ACTION").str.contains("Missed")
                ).alias("is_made"),
                pl.col("ACTION").str.contains("Two Pointer").alias("is_2pt"),
                pl.col("ACTION").str.contains("Three Pointer").alias("is_3pt"),
                (
                    pl.col("ACTION").str.contains("Two Pointer|Three Pointer")
                    & ~pl.col("ACTION").str.contains("Missed")
                ).alias("is_fg_made"),
                # Shot distance (Euclidean distance from basket)
                (
                    pl.col("COORD_X").cast(pl.Float32).pow(2)
                    + pl.col("COORD_Y").cast(pl.Float32).pow(2)
                )
                .sqrt()
                .alias("shot_distance"),
                # Shot angle from basket (in radians)
                pl.arctan2(
                    pl.col("COORD_Y").cast(pl.Float32),
                    pl.col("COORD_X").cast(pl.Float32),
                ).alias("shot_angle"),
            ]
        )

        # Add shot quality indicators
        df = df.with_columns(
            [
                pl.when(pl.col("FASTBREAK") == 1)
                .then(1.3)  # Higher weight for fastbreak shots
                .when(pl.col("SECOND_CHANCE") == 1)
                .then(1.1)  # Moderate weight for second chance shots
                .when(pl.col("shot_distance") < 3)
                .then(1.2)  # Higher weight for close shots
                .when(pl.col("shot_distance") > 7)
                .then(0.8)  # Lower weight for long shots
                .otherwise(1.0)
                .alias("shot_quality"),
                # Add distance-based shot categories
                (pl.col("shot_distance") < 3).alias("is_close_shot"),
                ((pl.col("shot_distance") >= 3) & (pl.col("shot_distance") < 5)).alias(
                    "is_mid_range"
                ),
                (pl.col("shot_distance") >= 5).alias("is_long_range"),
                # Points scored
                pl.when(pl.col("is_made") & pl.col("is_2pt"))
                .then(2)
                .when(pl.col("is_made") & pl.col("is_3pt"))
                .then(3)
                .otherwise(0)
                .alias("points_scored"),
            ]
        )

        # Calculate features by player and game
        result = df.group_by(["Player_ID", "Gamecode"]).agg(
            [
                # Shot quality metrics
                pl.col("shot_quality").mean().alias("shot_quality_index"),
                pl.col("shot_distance").mean().alias("avg_shot_distance"),
                pl.col("shot_distance").std().alias("shot_distance_variation"),
                (pl.col("COORD_X").std() * pl.col("COORD_Y").std()).alias(
                    "court_coverage_area"
                ),
                pl.col("ZONE").n_unique().alias("unique_scoring_zones"),
                # Shot type rates
                pl.col("FASTBREAK").mean().alias("fastbreak_rate"),
                pl.col("SECOND_CHANCE").mean().alias("second_chance_rate"),
                pl.col("is_3pt").mean().alias("three_point_attempt_rate"),
                # Shot distribution metrics
                pl.col("shot_distance").quantile(0.75).alias("shot_distance_75th"),
                pl.col("shot_distance").quantile(0.25).alias("shot_distance_25th"),
                pl.col("shot_angle").std().alias("shot_angle_spread"),
                # Points per shot
                pl.col("points_scored").mean().alias("points_per_shot"),
                # Shot success by distance
                (pl.col("is_made") & pl.col("is_close_shot"))
                .sum()
                .truediv(pl.col("is_close_shot").sum().clip(1))
                .alias("close_shot_percentage"),
                (pl.col("is_made") & pl.col("is_mid_range"))
                .sum()
                .truediv(pl.col("is_mid_range").sum().clip(1))
                .alias("mid_range_percentage"),
                (pl.col("is_made") & pl.col("is_long_range"))
                .sum()
                .truediv(pl.col("is_long_range").sum().clip(1))
                .alias("long_range_percentage"),
                # Shot success in different situations
                (pl.col("is_made") & (pl.col("FASTBREAK") == 1))
                .sum()
                .truediv((pl.col("FASTBREAK") == 1).sum().clip(1))
                .alias("fastbreak_fg_percentage"),
                (pl.col("is_made") & (pl.col("SECOND_CHANCE") == 1))
                .sum()
                .truediv((pl.col("SECOND_CHANCE") == 1).sum().clip(1))
                .alias("second_chance_fg_percentage"),
                # Effective field goal percentage
                (
                    (pl.col("is_made") & pl.col("is_2pt")).sum()
                    + 1.5 * (pl.col("is_made") & pl.col("is_3pt")).sum()
                )
                .truediv((pl.col("is_2pt").sum() + pl.col("is_3pt").sum()).clip(1))
                .alias("effective_fg_percentage"),
            ]
        )

        # Fill any remaining nulls with 0
        result = result.fill_null(0)

        # Ensure effective_fg_percentage exists before renaming
        if "effective_fg_percentage" not in result.columns:
            result = result.with_columns([pl.lit(0.0).alias("effective_fg_percentage")])

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of features created by this transformer."""
        return [
            "shot_quality_index",
            "avg_shot_distance",
            "shot_distance_variation",
            "court_coverage_area",
            "unique_scoring_zones",
            "fastbreak_rate",
            "second_chance_rate",
            "three_point_attempt_rate",
            "shot_distance_75th",
            "shot_distance_25th",
            "shot_angle_spread",
            "points_per_shot",
            "effective_fg_percentage",
            "close_shot_percentage",
            "mid_range_percentage",
            "long_range_percentage",
            "fastbreak_fg_percentage",
            "second_chance_fg_percentage",
        ]


# Register transformer
registry.register("shot_patterns", ShotPatternsTransformer)
