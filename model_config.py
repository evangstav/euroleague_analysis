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
    "is_home"
]

TARGET_COLUMN = "next_PIR"
ID_COLUMNS = ["Season", "Phase", "Round", "Gamecode", "Player_ID"]


def prepare_features(df):
    """Common feature preparation logic."""
    # Sort by time for proper train/test splitting
    df = df.sort_values(["Season", "Round"])
    
    # Create target variable (next game PIR)
    df[TARGET_COLUMN] = df.groupby("Player_ID")["PIR"].shift(-1)
    
    # Remove rows where next_PIR is NA (end of season games)
    df = df.dropna(subset=[TARGET_COLUMN])
    
    # Handle missing values
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0).values
    
    return df
