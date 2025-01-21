"""
Shared configuration for model training and analysis.
"""

FEATURE_COLUMNS = [
    # Basic efficiency metrics
    "true_shooting_percentage",
    "points_per_shot_right",
    "free_throw_rate",
    "offensive_rebound_rate_right",
    "defensive_rebound_rate_right",
    "assist_rate_right",
    "steal_rate_right",
    "block_rate_right",
    "turnover_rate_right",
    "usage_rate_right",
    "game_score",
    "floor_percentage",
    # Shot pattern metrics
    "shot_quality_index",
    "avg_shot_distance",
    "shot_distance_variation",
    "court_coverage_area",
    "unique_scoring_zones",
    "shot_angle_spread",
    "effective_fg_percentage",
    "close_shot_percentage",
    "mid_range_percentage",
    "long_range_percentage",
    "fastbreak_fg_percentage",
    "second_chance_fg_percentage",
    # Play-by-play impact metrics
    "avg_play_impact",
    "total_play_impact",
    "clutch_play_impact",
    "best_impact_sequence",
    "worst_impact_sequence",
    "game_volatility",
    "close_game_time_ratio",
    # Game situation metrics
    "super_clutch_plays",
    "super_clutch_scores",
    "close_game_impact_plays",
    "consecutive_positive_plays",
    # Rolling averages (past performance)
    "minutes_ma3",
    "minutes_ma5",
    "minutes_ma10",
    "points_ma3",
    "points_ma5",
    "points_ma10",
    "pir_ma3",
    "pir_ma5",
    "pir_ma10",
    # Advanced rolling metrics
    "usage_rate_ma3",
    "usage_rate_ma5",
    "net_contribution_ma3",
    "net_contribution_ma5",
    "scoring_efficiency_ma3",
    "scoring_efficiency_ma5",
    # Consistency and form metrics
    "pir_median_ma5",
    "pir_std5",
    "consistency_score",
    "trend_indicator",
    "recent_form",
    "pir_vs_season_avg",
    # Exponentially weighted metrics
    "pir_ewm",
    "points_ewm",
    "net_contribution_ewm",
    # Current game basic stats
    "minutes_played",
    "Points",
    # Key efficiency metrics
    "ast_to_turnover",
    "fg_percentage",
]

TARGET_COLUMN = "next_PIR"
ID_COLUMNS = ["Season", "Phase", "Round", "Gamecode", "Player_ID"]


def prepare_features(df, for_prediction=False):
    """Common feature preparation logic.

    Works with both Polars and Pandas DataFrames.
    """
    # Convert Polars DataFrame to Pandas if needed
    is_polars = hasattr(df, "to_pandas")
    if is_polars:
        df = df.to_pandas()
    print(df.columns)
    if not for_prediction:
        # Sort by time for proper train/test splitting
        df = df.sort_values(["Season", "Round"])

        # Create target variable (next game PIR)
        df[TARGET_COLUMN] = df.groupby("Player_ID")["PIR"].shift(-1)

        # Remove rows where next_PIR is NA (end of season games)
        df = df.dropna(subset=[TARGET_COLUMN])

    # Create a mapping of actual column names to expected names
    column_mapping = {}
    df_cols_lower = {col.lower(): col for col in df.columns}
    for feature in FEATURE_COLUMNS:
        if feature.lower() in df_cols_lower:
            column_mapping[feature] = df_cols_lower[feature.lower()]
        else:
            # Try common variations
            variations = [
                feature,
                feature.replace("_", ""),
                feature.title().replace("_", ""),
                feature.upper(),
            ]
            for var in variations:
                if var in df.columns:
                    column_mapping[feature] = var
                    break

    # Rename columns to match expected names
    df = df.rename(columns={v: k for k, v in column_mapping.items()})

    # Handle missing values
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    # For prediction, we don't need the target column
    if for_prediction:
        # Keep only the necessary columns
        keep_cols = FEATURE_COLUMNS + ID_COLUMNS + ["PIR"]  # Keep PIR for reference
        all_cols = set(df.columns)
        cols_to_keep = [col for col in keep_cols if col in all_cols]
        df = df[cols_to_keep]

    return df
