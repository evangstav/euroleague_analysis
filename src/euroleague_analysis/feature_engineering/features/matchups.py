"""Position matchup features."""

from typing import List

import polars as pl

from ..core.base import FeatureTransformer
from ..core.registry import registry


class PlayerMatchupTransformer(FeatureTransformer):
    """Analyze player performance against other players."""

    def __init__(self):
        super().__init__(
            name="position_matchups",  # Keep same name for compatibility
            required_columns=[
                "Player_ID",
                "Gamecode",
                "PLAYTYPE",
                "POINTS_A",
                "POINTS_B",
            ],
        )

    def fit(self, df: pl.DataFrame) -> "PlayerMatchupTransformer":
        """No fitting required for matchup features."""
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate player matchup features.

        Args:
            df: Input DataFrame with play-by-play data

        Returns:
            DataFrame with matchup features
        """
        self.validate_columns(df)

        # Add helper columns for scoring plays
        df = df.with_columns(
            [
                # Scoring plays
                pl.col("PLAYTYPE").str.contains("2FGM|3FGM|FTM").alias("is_score"),
                # Points scored (extract from playtype)
                pl.when(pl.col("PLAYTYPE").str.contains("2FGM"))
                .then(pl.lit(2))
                .when(pl.col("PLAYTYPE").str.contains("3FGM"))
                .then(pl.lit(3))
                .when(pl.col("PLAYTYPE").str.contains("FTM"))
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias("points_scored"),
                # Score differential
                (
                    pl.col("POINTS_A").cast(pl.Float32)
                    - pl.col("POINTS_B").cast(pl.Float32)
                )
                .abs()
                .alias("score_diff"),
            ]
        )

        # Calculate matchup features by player and game
        result = df.group_by(["Player_ID", "Gamecode"]).agg(
            [
                (
                    pl.col("points_scored").sum()
                    / pl.col("PLAYTYPE").str.contains("2FGA|3FGA|FTA").sum()
                ).alias("scoring_efficiency"),
                # Performance in close games
                (pl.col("points_scored").filter(pl.col("score_diff") <= 5).sum()).alias(
                    "points_in_close_games"
                ),
                # Clutch scoring (4th quarter)
                (
                    pl.col("points_scored")
                    .filter((pl.col("PERIOD") == 4) & (pl.col("MINUTE") >= 35))
                    .sum()
                ).alias("clutch_points"),
                # Defensive impact
                pl.col("PLAYTYPE")
                .str.contains("BLK|ST")
                .sum()
                .alias("defensive_plays"),
                # Offensive consistency
                pl.col("points_scored").std().alias("scoring_volatility"),
                # Game impact
                (
                    pl.col("points_scored").sum() * 100.0 / pl.col("PLAYTYPE").count()
                ).alias("usage_rate"),
            ]
        )

        # Fill nulls with 0
        result = result.with_columns(
            [
                pl.col(col).fill_null(0)
                for col in [
                    "scoring_efficiency",
                    "points_in_close_games",
                    "clutch_points",
                    "defensive_plays",
                    "scoring_volatility",
                    "usage_rate",
                ]
            ]
        )

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of features created by this transformer."""
        return [
            "scoring_efficiency",
            "points_in_close_games",
            "clutch_points",
            "defensive_plays",
            "scoring_volatility",
            "usage_rate",
        ]


# Register transformer
registry.register("position_matchups", PlayerMatchupTransformer)
