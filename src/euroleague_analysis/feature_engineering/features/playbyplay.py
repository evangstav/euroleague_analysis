"""Play-by-play features."""

from typing import List

import polars as pl

from ..core.base import FeatureTransformer
from ..core.registry import registry


class PlayByPlayTransformer(FeatureTransformer):
    """Extract features from play-by-play data."""

    def __init__(self):
        super().__init__(
            name="playbyplay",
            required_columns=[
                "Player_ID",
                "Gamecode",
                "PERIOD",
                "MINUTE",
                "PLAYTYPE",
                "POINTS_A",
                "POINTS_B",
                "NUMBEROFPLAY",
            ],
        )

    def fit(self, df: pl.DataFrame) -> "PlayByPlayTransformer":
        """No fitting required for play-by-play features."""
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate play-by-play features.

        Args:
            df: Input DataFrame with play-by-play data

        Returns:
            DataFrame with play-by-play features
        """
        self.validate_columns(df)

        # Add helper columns
        df = df.with_columns(
            [
                # Game time (normalized to 0-1 range)
                ((pl.col("PERIOD") - 1) * 10 + pl.col("MINUTE") - 30).alias(
                    "game_time"
                ),
                # Clutch time indicators with different thresholds
                ((pl.col("PERIOD") == 4) & (pl.col("MINUTE") >= 35)).alias("is_clutch"),
                ((pl.col("PERIOD") == 4) & (pl.col("MINUTE") >= 38)).alias(
                    "is_super_clutch"
                ),
                # Score differential and game situation
                (
                    pl.col("POINTS_A").cast(pl.Float32)
                    - pl.col("POINTS_B").cast(pl.Float32)
                ).alias("score_diff"),
                # Play type indicators with more granularity
                pl.col("PLAYTYPE").str.contains("AS").alias("is_assist"),
                pl.col("PLAYTYPE").str.contains("[23]FGA").alias("is_shot_attempt"),
                pl.col("PLAYTYPE").str.contains("D|BLK|ST").alias("is_defensive_play"),
                pl.col("PLAYTYPE").str.contains("TO").alias("is_turnover"),
                pl.col("PLAYTYPE").str.contains("O").alias("is_offensive_rebound"),
                pl.col("PLAYTYPE").str.contains("D").alias("is_defensive_rebound"),
                pl.col("PLAYTYPE").str.contains("BLK").alias("is_block"),
                pl.col("PLAYTYPE").str.contains("ST").alias("is_steal"),
                # Made shots/plays
                pl.col("PLAYTYPE").str.contains("Made|FTM").alias("is_made"),
                # High-impact plays
                pl.col("PLAYTYPE")
                .is_in(["2FGM", "3FGM", "FTM", "D", "O", "AS", "ST", "BLK"])
                .alias("is_positive"),
                # Game situation indicators
                (
                    pl.col("POINTS_A").cast(pl.Float32)
                    - pl.col("POINTS_B").cast(pl.Float32)
                )
                .abs()
                .le(5)
                .alias("is_close"),
                # Play impact score
                pl.when(pl.col("PLAYTYPE").str.contains("3FGM"))
                .then(4.0)  # 3pt + potential assist
                .when(pl.col("PLAYTYPE").str.contains("2FGM"))
                .then(3.0)  # 2pt + potential assist
                .when(pl.col("PLAYTYPE").str.contains("AS"))
                .then(2.0)  # Direct assist
                .when(pl.col("PLAYTYPE").str.contains("ST"))
                .then(2.0)  # Steal (possession change)
                .when(pl.col("PLAYTYPE").str.contains("BLK"))
                .then(1.5)  # Block
                .when(pl.col("PLAYTYPE").str.contains("O"))
                .then(1.2)  # Offensive rebound
                .when(pl.col("PLAYTYPE").str.contains("D"))
                .then(1.0)  # Defensive rebound
                .when(pl.col("PLAYTYPE").str.contains("TO"))
                .then(-1.5)  # Turnover
                .otherwise(0.0)
                .alias("play_impact"),
            ]
        )

        # Calculate features by player and game
        result = df.group_by(["Player_ID", "Gamecode"]).agg(
            [
                # Clutch performance with different thresholds
                pl.col("is_clutch").sum().alias("clutch_plays"),
                (pl.col("is_clutch") & pl.col("is_made")).sum().alias("clutch_scores"),
                pl.col("is_super_clutch").sum().alias("super_clutch_plays"),
                (pl.col("is_super_clutch") & pl.col("is_made"))
                .sum()
                .alias("super_clutch_scores"),
                # Game flow contribution
                (pl.col("is_close") & pl.col("is_positive"))
                .sum()
                .alias("close_game_impact_plays"),
                # Period-specific activity
                (pl.col("PERIOD") == 1).sum().alias("first_quarter_plays"),
                (pl.col("PERIOD") == 4).sum().alias("fourth_quarter_plays"),
                # Play diversity and versatility
                pl.col("PLAYTYPE").n_unique().alias("unique_play_types"),
                # Advanced play rates
                (pl.col("is_assist").sum() * 100.0 / pl.col("PLAYTYPE").count()).alias(
                    "assist_rate"
                ),
                (
                    pl.col("is_shot_attempt").sum() * 100.0 / pl.col("PLAYTYPE").count()
                ).alias("shot_attempt_rate"),
                (
                    pl.col("is_defensive_play").sum()
                    * 100.0
                    / pl.col("PLAYTYPE").count()
                ).alias("defensive_play_rate"),
                (
                    pl.col("is_turnover").sum()
                    * 100.0
                    / pl.col("is_shot_attempt").add(pl.col("is_assist")).sum()
                ).alias("turnover_rate"),
                # Play type specific rates
                (
                    pl.col("is_offensive_rebound").sum()
                    * 100.0
                    / pl.col("PLAYTYPE").count()
                ).alias("offensive_rebound_rate"),
                (
                    pl.col("is_defensive_rebound").sum()
                    * 100.0
                    / pl.col("PLAYTYPE").count()
                ).alias("defensive_rebound_rate"),
                (pl.col("is_block").sum() * 100.0 / pl.col("PLAYTYPE").count()).alias(
                    "block_rate"
                ),
                (pl.col("is_steal").sum() * 100.0 / pl.col("PLAYTYPE").count()).alias(
                    "steal_rate"
                ),
                # Impact metrics
                pl.col("play_impact").mean().alias("avg_play_impact"),
                pl.col("play_impact").sum().alias("total_play_impact"),
                (pl.col("play_impact") * pl.col("is_clutch"))
                .sum()
                .alias("clutch_play_impact"),
                # Game flow metrics
                pl.col("score_diff").std().alias("game_volatility"),
                (pl.col("score_diff").abs() <= 5).mean().alias("close_game_time_ratio"),
            ]
        )

        # Calculate momentum and streaks
        consecutive = (
            df.sort(["Player_ID", "Gamecode", "NUMBEROFPLAY"])
            .with_columns(
                [
                    # Track consecutive positive plays
                    pl.col("is_positive")
                    .shift(-1)
                    .over(["Player_ID", "Gamecode"])
                    .alias("next_is_positive"),
                    # Track play impact streaks
                    pl.col("play_impact")
                    .rolling_sum(window_size=3, center=False)
                    .over(["Player_ID", "Gamecode"])
                    .alias("rolling_impact"),
                ]
            )
            .group_by(["Player_ID", "Gamecode"])
            .agg(
                [
                    (pl.col("is_positive") & pl.col("next_is_positive"))
                    .sum()
                    .alias("consecutive_positive_plays"),
                    pl.col("rolling_impact").max().alias("best_impact_sequence"),
                    pl.col("rolling_impact").min().alias("worst_impact_sequence"),
                ]
            )
        )

        # Combine features and handle nulls
        result = result.join(
            consecutive, on=["Player_ID", "Gamecode"], how="left"
        ).with_columns(
            [
                pl.col(col).fill_null(0)
                for col in [
                    "assist_rate",
                    "shot_attempt_rate",
                    "defensive_play_rate",
                    "turnover_rate",
                    "offensive_rebound_rate",
                    "defensive_rebound_rate",
                    "block_rate",
                    "steal_rate",
                    "avg_play_impact",
                    "total_play_impact",
                    "clutch_play_impact",
                    "game_volatility",
                    "close_game_time_ratio",
                    "consecutive_positive_plays",
                    "best_impact_sequence",
                    "worst_impact_sequence",
                ]
            ]
        )

        return result

    def get_feature_names(self) -> List[str]:
        """Get list of features created by this transformer."""
        return [
            "clutch_plays",
            "clutch_scores",
            "super_clutch_plays",
            "super_clutch_scores",
            "close_game_impact_plays",
            "first_quarter_plays",
            "fourth_quarter_plays",
            "unique_play_types",
            "assist_rate",
            "shot_attempt_rate",
            "defensive_play_rate",
            "turnover_rate",
            "offensive_rebound_rate",
            "defensive_rebound_rate",
            "block_rate",
            "steal_rate",
            "avg_play_impact",
            "total_play_impact",
            "clutch_play_impact",
            "game_volatility",
            "close_game_time_ratio",
            "consecutive_positive_plays",
            "best_impact_sequence",
            "worst_impact_sequence",
        ]


# Register transformer
registry.register("playbyplay", PlayByPlayTransformer)
