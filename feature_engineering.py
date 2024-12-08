"""
Feature engineering module for PIR prediction using DuckDB.
"""

from pathlib import Path
import logging
import yaml
import json
import duckdb
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load parameters from params.yaml"""
    with open(params_path) as f:
        params = yaml.safe_load(f).get("features", {})
    return params


def init_duckdb(db_path: str) -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB connection with custom functions"""
    conn = duckdb.connect(db_path)

    conn.execute("""
        CREATE MACRO convert_minutes(time_str) AS (
            CASE 
                WHEN time_str = '' OR time_str IS NULL OR time_str = 'DNP' THEN 0
                ELSE CAST(SPLIT_PART(time_str, ':', 1) AS INTEGER) + 
                     CAST(SPLIT_PART(time_str, ':', 2) AS FLOAT) / 60
            END
        );
    """)

    return conn


def create_rolling_stats_view(
    conn: duckdb.DuckDBPyConnection, season: int, data_dir: Path
):
    """Create view with rolling statistics from box score data"""
    logger.info("Creating rolling statistics view")

    conn.execute(f"""
        CREATE OR REPLACE VIEW player_stats_features AS
        WITH player_stats AS (
            SELECT 
                *,
                convert_minutes(Minutes) as minutes_float
            FROM read_parquet('{data_dir}/raw/player_stats_{season}.parquet')
        ),
        game_stats AS (
            SELECT 
                Season,
                Phase,
                Round,
                Gamecode,
                Player_ID,
                -- Convert minutes from mm:ss to float
                convert_minutes(Minutes) as minutes_played,
                IsStarter::BOOLEAN as is_starter,
                Home::BOOLEAN as is_home,
                
                -- Basic stats
                Points,
                FieldGoalsMade2,
                FieldGoalsAttempted2,
                FieldGoalsMade3,
                FieldGoalsAttempted3,
                FreeThrowsMade,
                FreeThrowsAttempted,
                OffensiveRebounds,
                DefensiveRebounds,
                TotalRebounds,
                Assistances,
                Steals,
                Turnovers,
                BlocksFavour,
                BlocksAgainst,
                FoulsCommited,
                FoulsReceived,
                Plusminus,
                
                -- Calculated PIR components
                (Points + TotalRebounds + Assistances + Steals + BlocksFavour + FoulsReceived) as positive_actions,
                (FieldGoalsAttempted2 - FieldGoalsMade2 + FieldGoalsAttempted3 - FieldGoalsMade3 + 
                FreeThrowsAttempted - FreeThrowsMade + Turnovers + BlocksAgainst + FoulsCommited) as negative_actions,
                
                -- Calculate PIR
                (Points + TotalRebounds + Assistances + Steals + BlocksFavour + FoulsReceived) - 
                (FieldGoalsAttempted2 - FieldGoalsMade2 + FieldGoalsAttempted3 - FieldGoalsMade3 + 
                FreeThrowsAttempted - FreeThrowsMade + Turnovers + BlocksAgainst + FoulsCommited) as PIR,
                
                -- Efficiency metrics
                CAST(Points as FLOAT) / NULLIF(convert_minutes(Minutes), 0) as points_per_minute,
                CAST(FieldGoalsMade2 + FieldGoalsMade3 as FLOAT) / NULLIF(FieldGoalsAttempted2 + FieldGoalsAttempted3, 0) as fg_percentage,
                CAST(FieldGoalsMade2 as FLOAT) / NULLIF(FieldGoalsAttempted2, 0) as fg_percentage_2pt,
                CAST(FieldGoalsMade3 as FLOAT) / NULLIF(FieldGoalsAttempted3, 0) as fg_percentage_3pt,
                CAST(FreeThrowsMade as FLOAT) / NULLIF(FreeThrowsAttempted, 0) as ft_percentage,
                
                -- Shot distribution
                CAST(FieldGoalsAttempted3 as FLOAT) / NULLIF(FieldGoalsAttempted2 + FieldGoalsAttempted3, 0) as three_point_attempt_rate,
                
                -- Advanced metrics
                CAST(Assistances as FLOAT) / NULLIF(Turnovers, 0) as ast_to_turnover,
                CAST(OffensiveRebounds + DefensiveRebounds as FLOAT) / NULLIF(convert_minutes(Minutes), 0) as rebounds_per_minute,
                CAST(Steals + BlocksFavour as FLOAT) / NULLIF(convert_minutes(Minutes), 0) as defensive_plays_per_minute,
                
                -- Usage indicators
                FieldGoalsAttempted2 + FieldGoalsAttempted3 + FreeThrowsAttempted * 0.44 + Turnovers as possessions_used
                
            FROM player_stats
        ),

        -- Add rolling averages and game-by-game rankings
        player_features AS (
            SELECT 
                *,
                -- Rolling averages (3 games)
                AVG(minutes_played) OVER (
                    PARTITION BY Player_ID 
                    ORDER BY Season, Round 
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as minutes_ma3,
                
                AVG(Points) OVER (
                    PARTITION BY Player_ID 
                    ORDER BY Season, Round 
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as points_ma3,
                
                AVG(PIR) OVER (
                    PARTITION BY Player_ID 
                    ORDER BY Season, Round 
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as pir_ma3,
                
                AVG(fg_percentage) OVER (
                    PARTITION BY Player_ID 
                    ORDER BY Season, Round 
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as fg_percentage_ma3,
                
                AVG(possessions_used) OVER (
                    PARTITION BY Player_ID 
                    ORDER BY Season, Round 
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as usage_ma3,
                
                -- Rolling standard deviations (3 games) for consistency metrics
                STDDEV(PIR) OVER (
                    PARTITION BY Player_ID 
                    ORDER BY Season, Round 
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as pir_std3,
                
                -- Game-by-game rankings
                RANK() OVER (
                    PARTITION BY Gamecode 
                    ORDER BY PIR DESC
                ) as pir_rank_in_game,
                
                -- Difference from season average
                PIR - AVG(PIR) OVER (PARTITION BY Player_ID, Season) as pir_vs_season_avg,
                
                -- Form indicators
                CASE 
                    WHEN PIR > LAG(PIR) OVER (PARTITION BY Player_ID ORDER BY Season, Round) THEN 1
                    ELSE 0
                END as improving_form
                
            FROM game_stats
        )

        -- Final feature set
        SELECT 
            Season,
            Phase,
            Round,
            Gamecode,
            Player_ID,
            is_starter,
            is_home,
            minutes_played,
            PIR,
            
            -- Basic stats and efficiencies
            Points,
            fg_percentage,
            fg_percentage_2pt,
            fg_percentage_3pt,
            ft_percentage,
            three_point_attempt_rate,
            ast_to_turnover,
            rebounds_per_minute,
            defensive_plays_per_minute,
            
            -- Rolling averages
            minutes_ma3,
            points_ma3,
            pir_ma3,
            fg_percentage_ma3,
            usage_ma3,
            
            -- Consistency and form
            pir_std3,
            pir_rank_in_game,
            pir_vs_season_avg,
            improving_form,
            
            -- Raw counting stats for reference
            OffensiveRebounds,
            DefensiveRebounds,
            Assistances,
            Steals,
            Turnovers,
            BlocksFavour,
            BlocksAgainst,
            FoulsCommited,
            FoulsReceived,
            Plusminus,
            
            -- Components for transparency
            positive_actions,
            negative_actions
            
        FROM player_features
        WHERE minutes_played > 0  -- Filter out DNPs
        ORDER BY Season, Round, Gamecode, PIR DESC
    """)


def create_player_tier_view(conn: duckdb.DuckDBPyConnection):
    logger.info("Creating player tier view")
    conn.execute("""
        CREATE OR REPLACE VIEW player_tiers_view AS
        WITH player_season_stats AS (
            SELECT 
                Player_ID,
                COUNT(*) as games_played,
                AVG(PIR) as avg_season_pir,
                STDDEV(PIR) as std_season_pir,
                MAX(PIR) as max_season_pir,
                MIN(PIR) as min_season_pir,
                AVG(CASE WHEN PIR >= 0 THEN PIR END) as avg_positive_pir
            FROM player_stats_features
            GROUP BY Player_ID
        ),
        
        -- Calculate std deviation thresholds
        std_thresholds AS (
            SELECT 
                PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY std_season_pir) as std_33_percentile,
                PERCENTILE_CONT(0.67) WITHIN GROUP (ORDER BY std_season_pir) as std_67_percentile
            FROM player_season_stats
            WHERE games_played >= 10
        ),

        -- Create player tiers based on season performance
        player_tiers AS (
            SELECT 
                s.*,
                CASE
                    WHEN games_played >= 10 AND avg_season_pir >= 20 THEN 'elite'
                    WHEN games_played >= 10 AND avg_season_pir >= 15 THEN 'high_impact'
                    WHEN games_played >= 10 AND avg_season_pir >= 10 THEN 'solid'
                    WHEN games_played >= 10 THEN 'role_player'
                    ELSE 'insufficient_data'
                END as player_tier,
                
                CASE
                    WHEN games_played >= 10 AND std_season_pir <= (SELECT std_33_percentile FROM std_thresholds) THEN 'consistent'
                    WHEN games_played >= 10 AND std_season_pir >= (SELECT std_67_percentile FROM std_thresholds) THEN 'volatile'
                    WHEN games_played >= 10 THEN 'moderate'
                    ELSE 'insufficient_data'
                END as consistency_tier
            FROM player_season_stats s
        )

        SELECT 
            p.*,
            t.player_tier,
            t.consistency_tier,
            t.avg_season_pir,
            t.std_season_pir,
            t.games_played,
            t.avg_positive_pir,
            t.max_season_pir,
            t.min_season_pir
        FROM player_stats_features p
        LEFT JOIN player_tiers t ON p.Player_ID = t.Player_ID
    """)


def create_shot_patterns_view(
    conn: duckdb.DuckDBPyConnection, season: int, data_dir: Path
):
    """Create view with comprehensive shot pattern analysis"""
    logger.info("Creating shot patterns view")

    conn.execute(f"""
        CREATE OR REPLACE VIEW shot_patterns AS
        WITH game_shots AS (
            SELECT 
                Season,
                Gamecode,
                ID_PLAYER as Player_ID,  -- Aliasing to match other views
                -- Basic counting
                COUNT(*) as total_shots,
                SUM(POINTS) as total_points,
                COUNT(*) FILTER (WHERE POINTS > 0) as made_shots,
                
                -- Shot types
                COUNT(*) FILTER (WHERE POINTS = 2) as attempts_2pt,
                COUNT(*) FILTER (WHERE POINTS = 2 AND POINTS > 0) as made_2pt,
                COUNT(*) FILTER (WHERE POINTS = 3) as attempts_3pt,
                COUNT(*) FILTER (WHERE POINTS = 3 AND POINTS > 0) as made_3pt,
                
                -- Special play types
                COUNT(*) FILTER (WHERE FASTBREAK = 1) as fastbreak_attempts,
                SUM(POINTS) FILTER (WHERE FASTBREAK = 1) as fastbreak_points,
                COUNT(*) FILTER (WHERE SECOND_CHANCE = 1) as second_chance_attempts,
                SUM(POINTS) FILTER (WHERE SECOND_CHANCE = 1) as second_chance_points,
                COUNT(*) FILTER (WHERE POINTS_OFF_TURNOVER = 1) as turnover_play_attempts,
                SUM(POINTS) FILTER (WHERE POINTS_OFF_TURNOVER = 1) as points_off_turnover,
                
                -- Time distribution
                COUNT(*) FILTER (WHERE MINUTE <= 10) as shots_q1,
                COUNT(*) FILTER (WHERE MINUTE > 10 AND MINUTE <= 20) as shots_q2,
                COUNT(*) FILTER (WHERE MINUTE > 20 AND MINUTE <= 30) as shots_q3,
                COUNT(*) FILTER (WHERE MINUTE > 30) as shots_q4,
                
                -- Court zones
                COUNT(*) FILTER (WHERE ZONE = 'A') as shots_zone_a,
                COUNT(*) FILTER (WHERE ZONE = 'B') as shots_zone_b,
                COUNT(*) FILTER (WHERE ZONE = 'C') as shots_zone_c,
                COUNT(*) FILTER (WHERE ZONE = 'D') as shots_zone_d,
                COUNT(*) FILTER (WHERE ZONE = 'E') as shots_zone_e,
                COUNT(*) FILTER (WHERE ZONE = 'F') as shots_zone_f,
                COUNT(*) FILTER (WHERE ZONE = 'G') as shots_zone_g,
                COUNT(*) FILTER (WHERE ZONE = 'H') as shots_zone_h,
                COUNT(*) FILTER (WHERE ZONE = 'I') as shots_zone_i,
                COUNT(*) FILTER (WHERE ZONE = 'J') as shots_zone_j,
                
                -- Game context
                AVG(POINTS_B - POINTS_A) as avg_score_margin_during_shots,
                
                -- Court positioning (aggregate stats)
                AVG(COORD_X) as avg_shot_x,
                AVG(COORD_Y) as avg_shot_y,
                STDDEV(COORD_X) as spread_shot_x,
                STDDEV(COORD_Y) as spread_shot_y

            FROM read_parquet('{data_dir}/raw/shot_data_{season}.parquet')
            GROUP BY Season, Gamecode, ID_PLAYER
        ),

        -- Derive efficiency metrics
        game_efficiency AS (
            SELECT 
                *,
                -- Shooting percentages
                CAST(made_shots AS FLOAT) / NULLIF(total_shots, 0) as fg_percentage,
                CAST(made_2pt AS FLOAT) / NULLIF(attempts_2pt, 0) as fg_percentage_2pt,
                CAST(made_3pt AS FLOAT) / NULLIF(attempts_3pt, 0) as fg_percentage_3pt,
                
                -- Special plays efficiency
                CAST(fastbreak_points AS FLOAT) / NULLIF(fastbreak_attempts, 0) as fastbreak_ppp,
                CAST(second_chance_points AS FLOAT) / NULLIF(second_chance_attempts, 0) as second_chance_ppp,
                CAST(points_off_turnover AS FLOAT) / NULLIF(turnover_play_attempts, 0) as turnover_ppp,
                
                -- Shot distribution ratios
                CAST(attempts_3pt AS FLOAT) / NULLIF(total_shots, 0) as three_point_rate,
                CAST(fastbreak_attempts AS FLOAT) / NULLIF(total_shots, 0) as fastbreak_rate,
                CAST(second_chance_attempts AS FLOAT) / NULLIF(total_shots, 0) as second_chance_rate,
                
                -- Quarter distribution
                CAST(shots_q1 AS FLOAT) / NULLIF(total_shots, 0) as q1_shot_rate,
                CAST(shots_q2 AS FLOAT) / NULLIF(total_shots, 0) as q2_shot_rate,
                CAST(shots_q3 AS FLOAT) / NULLIF(total_shots, 0) as q3_shot_rate,
                CAST(shots_q4 AS FLOAT) / NULLIF(total_shots, 0) as q4_shot_rate,
                
                -- Zone preferences (normalized)
                CAST(shots_zone_a AS FLOAT) / NULLIF(total_shots, 0) as zone_a_rate,
                CAST(shots_zone_b AS FLOAT) / NULLIF(total_shots, 0) as zone_b_rate,
                CAST(shots_zone_c AS FLOAT) / NULLIF(total_shots, 0) as zone_c_rate,
                CAST(shots_zone_d AS FLOAT) / NULLIF(total_shots, 0) as zone_d_rate,
                CAST(shots_zone_e AS FLOAT) / NULLIF(total_shots, 0) as zone_e_rate,
                CAST(shots_zone_f AS FLOAT) / NULLIF(total_shots, 0) as zone_f_rate,
                CAST(shots_zone_g AS FLOAT) / NULLIF(total_shots, 0) as zone_g_rate,
                CAST(shots_zone_h AS FLOAT) / NULLIF(total_shots, 0) as zone_h_rate,
                CAST(shots_zone_i AS FLOAT) / NULLIF(total_shots, 0) as zone_i_rate,
                CAST(shots_zone_j AS FLOAT) / NULLIF(total_shots, 0) as zone_j_rate
            FROM game_shots
        )

        -- Rolling averages for key metrics
        SELECT 
            *,
            AVG(fg_percentage) OVER (
                PARTITION BY Player_ID 
                ORDER BY Season, Gamecode 
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) as fg_percentage_ma3,
            
            AVG(three_point_rate) OVER (
                PARTITION BY Player_ID 
                ORDER BY Season, Gamecode 
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) as three_point_rate_ma3,
            
            AVG(fastbreak_ppp) OVER (
                PARTITION BY Player_ID 
                ORDER BY Season, Gamecode 
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) as fastbreak_ppp_ma3,
            
            AVG(total_points) OVER (
                PARTITION BY Player_ID 
                ORDER BY Season, Gamecode 
                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
            ) as points_ma3
            
        FROM game_efficiency
        WHERE total_shots > 0
    """)


def create_game_context_view(
    conn: duckdb.DuckDBPyConnection, season: int, data_dir: Path
):
    """Create view with game context information"""
    logger.info("Creating game context view")

    conn.execute(f"""
        CREATE OR REPLACE VIEW game_context AS
        WITH quarter_data AS (
            SELECT *
            FROM read_parquet('{data_dir}/raw/quarter_data_{season}.parquet')
        ),

        game_quarters AS (
            SELECT 
                Season,
                Phase,
                Round,
                Gamecode,
                Team,
                -- Basic quarter scores
                COALESCE(Quarter1, 0) as q1_points,
                COALESCE(Quarter2, 0) as q2_points,
                COALESCE(Quarter3, 0) as q3_points,
                COALESCE(Quarter4, 0) as q4_points,
                -- Handle overtime quarters
                COALESCE(Extra1, 0) as ot1_points,
                COALESCE(Extra2, 0) as ot2_points,
                COALESCE(Extra3, 0) as ot3_points,
                COALESCE(Extra4, 0) as ot4_points,
                
                -- Calculate total points
                COALESCE(Quarter1, 0) + COALESCE(Quarter2, 0) + 
                COALESCE(Quarter3, 0) + COALESCE(Quarter4, 0) +
                COALESCE(Extra1, 0) + COALESCE(Extra2, 0) + 
                COALESCE(Extra3, 0) + COALESCE(Extra4, 0) as total_points,
                
                -- Flag overtime games
                CASE WHEN Extra1 IS NOT NULL THEN 1 ELSE 0 END as went_to_ot
            FROM quarter_data
        ),

        team_opponent_scores AS (
            SELECT 
                a.Season,
                a.Phase,
                a.Round,
                a.Gamecode,
                a.Team,
                
                -- Own quarters
                a.q1_points,
                a.q2_points,
                a.q3_points,
                a.q4_points,
                a.total_points,
                
                -- Opponent quarters (using self-join)
                b.q1_points as opp_q1_points,
                b.q2_points as opp_q2_points,
                b.q3_points as opp_q3_points,
                b.q4_points as opp_q4_points,
                b.total_points as opp_total_points,
                
                -- Quarter differentials
                a.q1_points - b.q1_points as q1_differential,
                a.q2_points - b.q2_points as q2_differential,
                a.q3_points - b.q3_points as q3_differential,
                a.q4_points - b.q4_points as q4_differential,
                
                -- Overtime info
                a.went_to_ot,
                a.ot1_points,
                b.ot1_points as opp_ot1_points
                
            FROM game_quarters a
            JOIN game_quarters b 
                ON a.Gamecode = b.Gamecode 
                AND a.Team != b.Team
        ),

        quarter_features AS (
            SELECT 
                *,
                -- Scoring patterns
                q1_points + q2_points as first_half_points,
                q3_points + q4_points as second_half_points,
                opp_q1_points + opp_q2_points as opp_first_half_points,
                opp_q3_points + opp_q4_points as opp_second_half_points,
                
                -- Derived metrics
                CASE 
                    WHEN q2_points > q1_points THEN 1 
                    ELSE 0 
                END as improved_q2,
                
                CASE 
                    WHEN q4_points > q3_points THEN 1 
                    ELSE 0 
                END as improved_q4,
                
                -- Identify best/worst quarters
                GREATEST(q1_points, q2_points, q3_points, q4_points) as best_quarter_points,
                LEAST(q1_points, q2_points, q3_points, q4_points) as worst_quarter_points,
                
                -- Quarter-by-quarter win/loss
                CASE WHEN q1_points > opp_q1_points THEN 1 ELSE 0 END as won_q1,
                CASE WHEN q2_points > opp_q2_points THEN 1 ELSE 0 END as won_q2,
                CASE WHEN q3_points > opp_q3_points THEN 1 ELSE 0 END as won_q3,
                CASE WHEN q4_points > opp_q4_points THEN 1 ELSE 0 END as won_q4,
                
                -- Rolling stats (3 games)
                AVG(total_points) OVER (
                    PARTITION BY Team 
                    ORDER BY Season, Round 
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as points_ma3,
                
                AVG(total_points - opp_total_points) OVER (
                    PARTITION BY Team 
                    ORDER BY Season, Round 
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as point_differential_ma3
                
            FROM team_opponent_scores
        )

        SELECT 
            Season,
            Phase,
            Round,
            Gamecode,
            Team,
            
            -- Raw quarter scores
            q1_points,
            q2_points,
            q3_points,
            q4_points,
            total_points,
            
            -- Half comparisons
            first_half_points,
            second_half_points,
            first_half_points - opp_first_half_points as first_half_differential,
            second_half_points - opp_second_half_points as second_half_differential,
            
            -- Quarter metrics
            best_quarter_points,
            worst_quarter_points,
            best_quarter_points - worst_quarter_points as quarter_variance,
            
            -- Quarter wins
            won_q1 + won_q2 + won_q3 + won_q4 as quarters_won,
            
            -- Scoring improvement indicators
            improved_q2,
            improved_q4,
            
            -- Overtime
            went_to_ot,
            ot1_points,
            CASE WHEN ot1_points > opp_ot1_points THEN 1 ELSE 0 END as won_ot,
            
            -- Rolling averages
            points_ma3,
            point_differential_ma3,
            
            -- Game flow metrics
            STDDEV(CAST(q1_differential AS DOUBLE)) OVER (
                PARTITION BY Gamecode, Team 
                ORDER BY ARRAY[1,2,3,4] 
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) as game_volatility,
            
            -- Additional context
            total_points > opp_total_points as won_game,
            ABS(total_points - opp_total_points) as margin_of_victory

        FROM quarter_features
        ORDER BY Season, Round, Gamecode, Team
    """)


def create_playbyplay_features_view(
    conn: duckdb.DuckDBPyConnection, season: int, data_dir: Path
):
    """Create view with play-by-play derived features"""
    logger.info("Creating play-by-play features view")

    conn.execute(f"""
        CREATE OR REPLACE VIEW playbyplay_features AS
        WITH pbp_data AS (
            SELECT *
            FROM read_parquet('{data_dir}/raw/playbyplay_data_{season}.parquet')
        ),
        pbp_base AS (
            SELECT 
                Season,
                Phase,
                Round,
                Gamecode,
                PLAYER_ID,
                PERIOD,
                MINUTE,
                PLAYTYPE,
                POINTS_A,
                POINTS_B,
                -- Calculate score margin at time of play
                POINTS_A - POINTS_B as score_margin,
                -- Flag crucial moments
                CASE 
                    WHEN PERIOD = 4 AND MINUTE >= 35 THEN 1 
                    ELSE 0 
                END as clutch_time
            FROM pbp_data
        ),

        player_game_plays AS (
            SELECT 
                Season,
                Gamecode,
                PLAYER_ID,
                
                -- Play type frequencies
                COUNT(*) as total_plays,
                COUNT(*) FILTER (WHERE PLAYTYPE = 'AS') as assists,
                COUNT(*) FILTER (WHERE PLAYTYPE = 'FV') as fouls_drawn,
                COUNT(*) FILTER (WHERE PLAYTYPE = 'FC') as fouls_committed,
                COUNT(*) FILTER (WHERE PLAYTYPE = 'RD') as def_rebounds,
                COUNT(*) FILTER (WHERE PLAYTYPE = 'RO') as off_rebounds,
                COUNT(*) FILTER (WHERE PLAYTYPE = 'T1') as free_throws_made,
                COUNT(*) FILTER (WHERE PLAYTYPE = 'BP') as blocks,
                COUNT(*) FILTER (WHERE PLAYTYPE = 'BR') as blocks_received,
                COUNT(*) FILTER (WHERE PLAYTYPE = 'P2') as made_2pt,
                COUNT(*) FILTER (WHERE PLAYTYPE = 'P3') as made_3pt,
                COUNT(*) FILTER (WHERE PLAYTYPE = 'IN') as subbed_in,
                COUNT(*) FILTER (WHERE PLAYTYPE = 'OUT') as subbed_out,

                -- Clutch time plays
                COUNT(*) FILTER (WHERE clutch_time = 1) as clutch_plays,
                COUNT(*) FILTER (WHERE clutch_time = 1 AND PLAYTYPE IN ('P2', 'P3', 'T1')) as clutch_points_plays,
                COUNT(*) FILTER (WHERE clutch_time = 1 AND PLAYTYPE = 'AS') as clutch_assists,
                
                -- Score margin context
                AVG(score_margin) as avg_score_margin_when_playing,
                COUNT(*) FILTER (WHERE ABS(score_margin) <= 5) as close_game_plays,
                COUNT(*) FILTER (WHERE score_margin < -5) as plays_when_trailing,
                COUNT(*) FILTER (WHERE score_margin > 5) as plays_when_leading,
                
                -- Time distribution
                MIN(MINUTE) as first_action_minute,
                MAX(MINUTE) as last_action_minute,
                COUNT(DISTINCT PERIOD) as periods_played,
                
                -- Quarter-by-quarter involvement
                COUNT(*) FILTER (WHERE PERIOD = 1) as q1_plays,
                COUNT(*) FILTER (WHERE PERIOD = 2) as q2_plays,
                COUNT(*) FILTER (WHERE PERIOD = 3) as q3_plays,
                COUNT(*) FILTER (WHERE PERIOD = 4) as q4_plays

            FROM pbp_base
            GROUP BY Season, Gamecode, PLAYER_ID
        ),

        -- Add derived metrics and rolling stats
        player_game_features AS (
            SELECT 
                *,
                -- Derived metrics
                CAST(clutch_plays AS FLOAT) / NULLIF(total_plays, 0) as clutch_play_rate,
                CAST(close_game_plays AS FLOAT) / NULLIF(total_plays, 0) as close_game_play_rate,
                last_action_minute - first_action_minute as game_involvement_span,
                
                -- Play distribution
                CAST(q1_plays AS FLOAT) / NULLIF(total_plays, 0) as q1_play_rate,
                CAST(q2_plays AS FLOAT) / NULLIF(total_plays, 0) as q2_play_rate,
                CAST(q3_plays AS FLOAT) / NULLIF(total_plays, 0) as q3_play_rate,
                CAST(q4_plays AS FLOAT) / NULLIF(total_plays, 0) as q4_play_rate,
                
                -- Rolling averages (3 games)
                AVG(total_plays) OVER (
                    PARTITION BY PLAYER_ID 
                    ORDER BY Season, Gamecode 
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as total_plays_ma3,
                
                AVG(clutch_plays) OVER (
                    PARTITION BY PLAYER_ID 
                    ORDER BY Season, Gamecode 
                    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                ) as clutch_plays_ma3,
                
                
            FROM player_game_plays
        )

        SELECT 
            Season,
            Gamecode,
            PLAYER_ID,
            
            -- Activity volume
            total_plays,
            clutch_plays,
            close_game_plays,
            
            -- Play type breakdown
            assists,
            fouls_drawn,
            fouls_committed,
            def_rebounds,
            off_rebounds,
            free_throws_made,
            blocks,
            blocks_received,
            made_2pt,
            made_3pt,
            
            -- Game context
            avg_score_margin_when_playing,
            plays_when_trailing,
            plays_when_leading,
            clutch_play_rate,
            close_game_play_rate,
            
            -- Time distribution
            game_involvement_span,
            periods_played,
            q1_play_rate,
            q2_play_rate,
            q3_play_rate,
            q4_play_rate,
            
            -- Rolling averages
            total_plays_ma3,
            clutch_plays_ma3,
            
            -- Substitutions
            subbed_in,
            subbed_out

        FROM player_game_features
        WHERE total_plays > 0
        ORDER BY Season, Gamecode, total_plays DESC
    """)


def create_final_feature_set(conn: duckdb.DuckDBPyConnection, output_dir: Path):
    """Combine all features and create final dataset"""
    logger.info("Creating final feature set")

    feature_query = """
        CREATE OR REPLACE VIEW final_features AS
    SELECT 
        -- Basic identifiers and game context
        r.Season,
        r.Phase,
        r.Round,
        r.Gamecode,
        r.Player_ID,
        r.is_starter,
        r.is_home,
        r.minutes_played,
        r.PIR,
        
        -- Basic stats and efficiencies
        r.Points,
        r.fg_percentage,
        r.fg_percentage_2pt,
        r.fg_percentage_3pt,
        r.ft_percentage,
        r.three_point_attempt_rate,
        r.ast_to_turnover,
        r.rebounds_per_minute,
        r.defensive_plays_per_minute,
        
        -- Rolling averages from player stats
        r.minutes_ma3,
        r.points_ma3,
        r.pir_ma3,
        r.fg_percentage_ma3,
        r.usage_ma3,
        
        -- Consistency and form
        r.pir_std3,
        r.pir_rank_in_game,
        r.pir_vs_season_avg,
        r.improving_form,
        
        -- Raw counting stats
        r.OffensiveRebounds,
        r.DefensiveRebounds,
        r.Assistances,
        r.Steals,
        r.Turnovers,
        r.BlocksFavour,
        r.BlocksAgainst,
        r.FoulsCommited,
        r.FoulsReceived,
        r.Plusminus,
        
        -- PIR components
        r.positive_actions,
        r.negative_actions,
        
        -- Shot patterns features
        s.fastbreak_ppp,
        s.second_chance_ppp,
        s.fastbreak_ppp_ma3,
        s.zone_a_rate,
        s.zone_b_rate,
        s.zone_c_rate,
        s.zone_d_rate,
        s.zone_e_rate,
        s.zone_f_rate,
        s.zone_g_rate,
        s.zone_h_rate,
        s.zone_i_rate,
        s.zone_j_rate,
        
        -- Play-by-play derived features
        p.clutch_plays,
        p.clutch_play_rate,
        p.close_game_play_rate,
        p.game_involvement_span,
        p.q1_play_rate,
        p.q2_play_rate,
        p.q3_play_rate,
        p.q4_play_rate,
        p.clutch_plays_ma3,
        
        -- Game context features
        g.q1_points,
        g.q2_points,
        g.q3_points,
        g.q4_points,
        g.total_points,
        g.first_half_points,
        g.second_half_points,
        g.first_half_differential,
        g.second_half_differential,
        g.quarter_variance,
        g.quarters_won,
        g.improved_q2,
        g.improved_q4,
        g.went_to_ot,
        g.won_ot,
        g.points_ma3,
        g.point_differential_ma3,
        g.game_volatility,
        g.won_game,
        g.margin_of_victory,

        -- Player tier features
        h.player_tier,
        h.consistency_tier,
        h.avg_season_pir,
        h.std_season_pir,
        h.games_played,
        h.avg_positive_pir,
        h.max_season_pir,
        h.min_season_pir

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

    conn.execute(feature_query)

    # Create output path
    output_path = output_dir / "features.parquet"

    # Save to parquet using Python API
    df = conn.execute("SELECT * FROM final_features").df()

    df.to_parquet(str(output_path))

    logger.info(f"Saved features to {output_path}")


def main():
    # Load parameters
    params = load_params()

    # season = params.get("season", 2023)
    season = 2023

    # data_dir = Path(params.get("data_dir", "euroleague_data"))
    data_dir = Path("euroleague_data")

    db_path = str(data_dir / "features.duckdb")

    # Initialize DuckDB connection
    conn = init_duckdb(db_path)

    # Create feature views
    create_rolling_stats_view(conn, season, data_dir)

    create_shot_patterns_view(conn, season, data_dir)

    create_game_context_view(conn, season, data_dir)

    create_playbyplay_features_view(conn, season, data_dir)

    create_player_tier_view(conn)

    # Generate final feature set
    create_final_feature_set(conn, data_dir)

    # Save feature metadata
    metadata = {
        "features_file": str(data_dir / "features.parquet"),
        "database_file": db_path,
    }

    with open("feature_outputs.json", "w") as f:
        json.dump(metadata, f)

    conn.close()


if __name__ == "__main__":
    main()

