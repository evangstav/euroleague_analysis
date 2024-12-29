"""
Shared configuration for model training and analysis.
"""

FEATURE_COLUMNS = [
    # Player performance metrics
    "PIR",
    "Points",
    "minutes_played",
    "fg_percentage",
    "ft_percentage",
    "three_point_attempt_rate",
    "ast_to_turnover",
    "rebounds_per_minute",
    "defensive_plays_per_minute",
    # Rolling averages
    "minutes_ma3",
    "points_ma3",
    "pir_ma3",
    "fg_percentage_ma3",
    "usage_ma3",
    # Form and consistency
    "pir_std3",
    "pir_rank_in_game",
    "pir_vs_season_avg",
    "improving_form",
    # Efficiency metrics
    "fastbreak_ppp",
    "second_chance_ppp",
    "fastbreak_ppp_ma3",
    # Shot zone preferences
    "zone_a_rate",
    "zone_b_rate",
    "zone_c_rate",
    "zone_d_rate",
    "zone_e_rate",
    "zone_f_rate",
    "zone_g_rate",
    "zone_h_rate",
    "zone_i_rate",
    "zone_j_rate",
    # Game involvement
    "clutch_play_rate",
    "close_game_play_rate",
    "game_involvement_span",
    "q1_play_rate",
    "q2_play_rate",
    "q3_play_rate",
    "q4_play_rate",
    # Team performance context
    "points_ma3",
    "point_differential_ma3",
    "game_volatility",
    "margin_of_victory",
    # Game context flags
    "is_starter",
    "is_home",
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
