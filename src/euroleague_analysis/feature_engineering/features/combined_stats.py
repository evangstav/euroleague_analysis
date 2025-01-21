"""Combined basic and rolling statistical features."""

from typing import List

import polars as pl

from ..core.base import FeatureTransformer
from ..core.registry import registry


class CombinedStatsTransformer(FeatureTransformer):
    """Transform raw stats and calculate rolling statistics."""

    def __init__(self, window_sizes: List[int] = [3, 5, 10]):
        super().__init__(
            name="combined_stats",
            required_columns=[
                # Basic stats columns
                "Minutes",
                "Points",
                "FieldGoalsMade2",
                "FieldGoalsAttempted2",
                "FieldGoalsMade3",
                "FieldGoalsAttempted3",
                "FreeThrowsMade",
                "FreeThrowsAttempted",
                "OffensiveRebounds",
                "DefensiveRebounds",
                "Assistances",
                "Steals",
                "Turnovers",
                "BlocksFavour",
                "BlocksAgainst",
                "FoulsCommited",
                "FoulsReceived",
                # Rolling stats identifiers
                "Season",
                "Round",
                "Player_ID",
                "Gamecode",
            ],
        )
        self.window_sizes = window_sizes

    def fit(self, df: pl.DataFrame) -> "CombinedStatsTransformer":
        """No fitting required."""
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform raw stats and calculate rolling statistics."""
        self.validate_columns(df)

        # First process basic stats
        df = df.with_columns(
            [
                # Clean minutes
                pl.when(
                    pl.col("Minutes")
                    .is_in(["", "DNP"])
                    .or_(pl.col("Minutes").is_null())
                )
                .then(pl.lit("0:00"))
                .otherwise(pl.col("Minutes"))
                .alias("Minutes_clean")
            ]
        )

        df = df.with_columns(
            [
                # Convert minutes to float
                (
                    pl.col("Minutes_clean").str.split(":").list.get(0).cast(pl.Float32)
                    + pl.col("Minutes_clean")
                    .str.split(":")
                    .list.get(1)
                    .cast(pl.Float32)
                    / 60.0
                ).alias("minutes_played")
            ]
        )

        # Calculate basic stats
        df = df.with_columns(
            [
                # Shooting percentages
                (
                    (pl.col("FieldGoalsMade2") + pl.col("FieldGoalsMade3"))
                    / (pl.col("FieldGoalsAttempted2") + pl.col("FieldGoalsAttempted3"))
                    .clip(lower_bound=1)
                    .fill_null(1)
                ).alias("fg_percentage"),
                (
                    pl.col("FieldGoalsMade2")
                    / pl.col("FieldGoalsAttempted2").clip(lower_bound=1).fill_null(1)
                ).alias("fg_percentage_2pt"),
                (
                    pl.col("FieldGoalsMade3")
                    / pl.col("FieldGoalsAttempted3").clip(lower_bound=1).fill_null(1)
                ).alias("fg_percentage_3pt"),
                (
                    pl.col("FreeThrowsMade")
                    / pl.col("FreeThrowsAttempted").clip(lower_bound=1).fill_null(1)
                ).alias("ft_percentage"),
                # Advanced metrics
                (
                    pl.col("Points")
                    / (
                        2
                        * (
                            pl.col("FieldGoalsAttempted2")
                            + pl.col("FieldGoalsAttempted3")
                            + 0.44 * pl.col("FreeThrowsAttempted")
                        )
                    ).clip(lower_bound=1)
                ).alias("true_shooting_percentage"),
                # PIR calculation
                (
                    (
                        pl.col("Points")
                        + pl.col("OffensiveRebounds")
                        + pl.col("DefensiveRebounds")
                        + pl.col("Assistances")
                        + pl.col("Steals")
                        + pl.col("BlocksFavour")
                        + pl.col("FoulsReceived")
                    )
                    - (
                        pl.col("FieldGoalsAttempted2")
                        - pl.col("FieldGoalsMade2")
                        + pl.col("FieldGoalsAttempted3")
                        - pl.col("FieldGoalsMade3")
                        + pl.col("FreeThrowsAttempted")
                        - pl.col("FreeThrowsMade")
                        + pl.col("Turnovers")
                        + pl.col("BlocksAgainst")
                        + pl.col("FoulsCommited")
                    )
                ).alias("PIR"),
            ]
        )

        # Sort data for rolling calculations
        df = df.sort(["Player_ID", "Season", "Round"])

        # Calculate rolling stats
        for window in self.window_sizes:
            suffix = f"_ma{window}"

            df = df.with_columns(
                [
                    # Basic rolling averages
                    pl.col("minutes_played")
                    .rolling_mean(window_size=window, center=False)
                    .over("Player_ID")
                    .alias(f"minutes{suffix}"),
                    pl.col("Points")
                    .rolling_mean(window_size=window, center=False)
                    .over("Player_ID")
                    .alias(f"Points{suffix}"),
                    pl.col("PIR")
                    .rolling_mean(window_size=window, center=False)
                    .over("Player_ID")
                    .alias(f"pir{suffix}"),
                    # Rolling standard deviation
                    pl.col("PIR")
                    .rolling_std(window_size=window, center=False)
                    .over("Player_ID")
                    .alias(f"pir_std{suffix}"),
                ]
            )

        # Add weighted moving averages
        df = df.with_columns(
            [
                pl.col("PIR").ewm_mean(alpha=0.3).over("Player_ID").alias("pir_ewm"),
                pl.col("Points")
                .ewm_mean(alpha=0.3)
                .over("Player_ID")
                .alias("points_ewm"),
            ]
        )

        # Add form indicators
        df = df.with_columns(
            [(pl.col("pir_ma3") - pl.col("pir_ma10")).alias("recent_form")]
        )

        # Select final columns
        computed_features = [
            "minutes_played",
            "Points",
            "fg_percentage",
            "fg_percentage_2pt",
            "fg_percentage_3pt",
            "ft_percentage",
            "true_shooting_percentage",
            "PIR",
            "pir_ewm",
            "points_ewm",
            "recent_form",
        ]

        # Add rolling features
        for window in self.window_sizes:
            suffix = f"_ma{window}"
            computed_features.extend(
                [
                    f"minutes{suffix}",
                    f"Points{suffix}",
                    f"pir{suffix}",
                    f"pir_std{suffix}",
                ]
            )

        # Keep original columns needed by other transformers
        original_cols = [
            "Season",
            "Phase",
            "Round",
            "Gamecode",
            "Player_ID",
            "IsStarter",
            "Home",
        ]

        return df.select(original_cols + computed_features)

    def get_feature_names(self) -> List[str]:
        """Get list of features created by this transformer."""
        features = [
            "minutes_played",
            "Points",
            "fg_percentage",
            "fg_percentage_2pt",
            "fg_percentage_3pt",
            "ft_percentage",
            "true_shooting_percentage",
            "PIR",
            "pir_ewm",
            "points_ewm",
            "recent_form",
        ]

        for window in self.window_sizes:
            suffix = f"_ma{window}"
            features.extend(
                [
                    f"minutes{suffix}",
                    f"Points{suffix}",
                    f"pir{suffix}",
                    f"pir_std{suffix}",
                ]
            )

        return features


# Register transformer
registry.register("combined_stats", CombinedStatsTransformer)
