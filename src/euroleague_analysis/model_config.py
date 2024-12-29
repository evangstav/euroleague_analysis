"""
Shared configuration for model training and analysis.
"""

FEATURE_COLUMNS = [
    # Rolling averages (past performance)
    "minutes_ma3",
    "points_ma3",
    "pir_ma3",
    # Current game basic stats
    "minutes_played",
    "Points",
    # Shooting efficiency
    "fg_percentage",
    "ft_percentage",
    "fg_percentage_2pt",
    "fg_percentage_3pt",
    "three_point_rate",
    # Game situation stats
    "fastbreak_rate",
    "second_chance_rate",
    "clutch_plays",
    "clutch_scores",
    # Game involvement
    "first_quarter_plays",
    "fourth_quarter_plays",
    "close_game_plays",
    "consecutive_positive_plays",
    # Play style metrics
    "assist_rate",
    "shot_attempt_rate",
    "defensive_play_rate",
    "ast_to_turnover",
    "turnover_rate",
    # Form indicators
    "improving_form",
    "pir_std3",
    "pir_vs_season_avg",
    "pir_rank_in_game",
    # Game context
    "is_starter",
    "is_home",
]

TARGET_COLUMN = "next_PIR"
ID_COLUMNS = ["Season", "Phase", "Round", "Gamecode", "Player_ID"]


def prepare_features(df, for_prediction=False):
    """Common feature preparation logic."""
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
