CREATE OR REPLACE VIEW position_matchup_features_{season} AS
WITH player_style_stats AS (
    SELECT 
        ps.Player_ID,
        ps.Season,
        -- Shooting patterns
        SUM(CAST(ps.FieldGoalsAttempted3 as FLOAT)) / NULLIF(SUM(ps.FieldGoalsAttempted2 + ps.FieldGoalsAttempted3), 0) as three_point_rate,
        -- Rebounding patterns
        SUM(CAST(ps.TotalRebounds as FLOAT)) / NULLIF(SUM(convert_minutes(ps.Minutes)), 0) as rebounds_per_minute,
        -- Assist patterns
        SUM(CAST(ps.Assistances as FLOAT)) / NULLIF(SUM(convert_minutes(ps.Minutes)), 0) as assists_per_minute,
        -- Inside scoring
        SUM(CAST(ps.FieldGoalsMade2 as FLOAT)) / NULLIF(SUM(convert_minutes(ps.Minutes)), 0) as inside_scoring_rate,
        -- Shot location patterns from shot data
        AVG(CAST(s.COORD_Y as FLOAT)) as avg_shot_distance_from_basket
    FROM read_parquet('{data_dir}/raw/player_stats_{season}.parquet') ps
    LEFT JOIN read_parquet('{data_dir}/raw/shot_data_{season}.parquet') s 
        ON ps.Player_ID = s.ID_PLAYER AND ps.Gamecode = s.Gamecode
    WHERE ps.Player_ID != 'Total'
    GROUP BY ps.Player_ID, ps.Season
),

inferred_positions AS (
    SELECT 
        *,
        CASE
            WHEN three_point_rate > 0.4 AND assists_per_minute > 0.15 THEN 'G'  -- Guards: high 3pt rate, high assists
            WHEN rebounds_per_minute > 0.3 AND inside_scoring_rate > 0.15 THEN 'C'  -- Centers: high rebounds, inside scoring
            ELSE 'F'  -- Forwards: balanced stats
        END as inferred_position
    FROM player_style_stats
),

game_matchups AS (
    SELECT 
        g1.Season,
        g1.Phase,
        g1.Round,
        g1.Gamecode,
        g1.Player_ID,
        -- Calculate PIR for player
        (g1.Points + g1.TotalRebounds + g1.Assistances + g1.Steals + g1.BlocksFavour + g1.FoulsReceived) -
        (g1.FieldGoalsAttempted2 - g1.FieldGoalsMade2 + g1.FieldGoalsAttempted3 - g1.FieldGoalsMade3 +
        g1.FreeThrowsAttempted - g1.FreeThrowsMade + g1.Turnovers + g1.BlocksAgainst + g1.FoulsCommited) as player_pir,
        g1.Points as player_points,
        convert_minutes(g1.Minutes) as player_minutes,
        p1.inferred_position as player_position,
        g2.Player_ID as opponent_id,
        -- Calculate PIR for opponent
        (g2.Points + g2.TotalRebounds + g2.Assistances + g2.Steals + g2.BlocksFavour + g2.FoulsReceived) -
        (g2.FieldGoalsAttempted2 - g2.FieldGoalsMade2 + g2.FieldGoalsAttempted3 - g2.FieldGoalsMade3 +
        g2.FreeThrowsAttempted - g2.FreeThrowsMade + g2.Turnovers + g2.BlocksAgainst + g2.FoulsCommited) as opponent_pir,
        g2.Points as opponent_points,
        convert_minutes(g2.Minutes) as opponent_minutes,
        p2.inferred_position as opponent_position
    FROM read_parquet('{data_dir}/raw/player_stats_{season}.parquet') g1
    JOIN inferred_positions p1 ON g1.Player_ID = p1.Player_ID AND g1.Season = p1.Season
    -- Self-join to get opponent stats from same game
    JOIN read_parquet('{data_dir}/raw/player_stats_{season}.parquet') g2 
        ON g1.Gamecode = g2.Gamecode 
        AND g1.Player_ID != g2.Player_ID
        AND ((g1.Home::BOOLEAN AND NOT g2.Home::BOOLEAN) OR (NOT g1.Home::BOOLEAN AND g2.Home::BOOLEAN))
    JOIN inferred_positions p2 ON g2.Player_ID = p2.Player_ID AND g2.Season = p2.Season
    WHERE g1.Player_ID != 'Total' AND g2.Player_ID != 'Total'
),

position_stats AS (
    SELECT
        m.Season,
        m.Phase,
        m.Round,
        m.Gamecode,
        m.Player_ID,
        m.player_position,

        -- Overall matchup stats
        AVG(m.player_pir) as avg_pir,
        AVG(m.player_points) as avg_points,
        AVG(m.player_minutes) as avg_minutes,
        
        -- Stats vs Guards
        AVG(CASE WHEN m.opponent_position = 'G' THEN m.player_pir END) as avg_pir_vs_guards,
        AVG(CASE WHEN m.opponent_position = 'G' THEN m.player_points END) as avg_points_vs_guards,
        COUNT(CASE WHEN m.opponent_position = 'G' AND m.player_pir > m.opponent_pir THEN 1 END) as wins_vs_guards,
        
        -- Stats vs Forwards
        AVG(CASE WHEN m.opponent_position = 'F' THEN m.player_pir END) as avg_pir_vs_forwards,
        AVG(CASE WHEN m.opponent_position = 'F' THEN m.player_points END) as avg_points_vs_forwards,
        COUNT(CASE WHEN m.opponent_position = 'F' AND m.player_pir > m.opponent_pir THEN 1 END) as wins_vs_forwards,
        
        -- Stats vs Centers
        AVG(CASE WHEN m.opponent_position = 'C' THEN m.player_pir END) as avg_pir_vs_centers,
        AVG(CASE WHEN m.opponent_position = 'C' THEN m.player_points END) as avg_points_vs_centers,
        COUNT(CASE WHEN m.opponent_position = 'C' AND m.player_pir > m.opponent_pir THEN 1 END) as wins_vs_centers,
        
        -- Matchup efficiency (PIR differential)
        AVG(m.player_pir - m.opponent_pir) as avg_pir_differential,
        
        -- Position versatility (how well they perform against different positions)
        STDDEV(CASE 
            WHEN m.opponent_position IN ('G', 'F', 'C') 
            THEN m.player_pir 
        END) as position_matchup_volatility,
        
        -- Favorable matchup indicator
        MAX(CASE
            WHEN m.opponent_position = 'G' THEN AVG(m.player_pir - m.opponent_pir)
            WHEN m.opponent_position = 'F' THEN AVG(m.player_pir - m.opponent_pir)
            WHEN m.opponent_position = 'C' THEN AVG(m.player_pir - m.opponent_pir)
        END) OVER (PARTITION BY m.Player_ID) as best_matchup_differential,
        
        -- Rolling matchup performance (last 3 games)
        AVG(m.player_pir - m.opponent_pir) OVER (
            PARTITION BY m.Player_ID, m.opponent_position 
            ORDER BY m.Season, m.Round 
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as recent_matchup_performance
        
    FROM game_matchups m
    GROUP BY
        m.Season,
        m.Phase,
        m.Round,
        m.Gamecode,
        m.Player_ID,
        m.player_position,
        m.opponent_position,
        m.player_pir,
        m.opponent_pir,
        m.player_points,
        m.player_minutes,
        m.opponent_points,
        m.opponent_minutes
)

SELECT 
    s.*,
    -- Normalize performance vs each position relative to overall average
    s.avg_pir_vs_guards - s.avg_pir as guard_matchup_edge,
    s.avg_pir_vs_forwards - s.avg_pir as forward_matchup_edge,
    s.avg_pir_vs_centers - s.avg_pir as center_matchup_edge,
    
    -- Position dominance scores (wins vs specific position / total games vs that position)
    CAST(s.wins_vs_guards as FLOAT) / NULLIF(COUNT(*) FILTER (WHERE s.avg_pir_vs_guards IS NOT NULL), 0) as guard_dominance_rate,
    CAST(s.wins_vs_forwards as FLOAT) / NULLIF(COUNT(*) FILTER (WHERE s.avg_pir_vs_forwards IS NOT NULL), 0) as forward_dominance_rate,
    CAST(s.wins_vs_centers as FLOAT) / NULLIF(COUNT(*) FILTER (WHERE s.avg_pir_vs_centers IS NOT NULL), 0) as center_dominance_rate
FROM position_stats s
GROUP BY
    s.Season,
    s.Phase,
    s.Round,
    s.Gamecode,
    s.Player_ID,
    s.player_position,
    s.avg_pir,
    s.avg_points,
    s.avg_minutes,
    s.avg_pir_vs_guards,
    s.avg_points_vs_guards,
    s.avg_pir_vs_forwards,
    s.avg_points_vs_forwards,
    s.avg_pir_vs_centers,
    s.avg_points_vs_centers,
    s.wins_vs_guards,
    s.wins_vs_forwards,
    s.wins_vs_centers,
    s.avg_pir_differential,
    s.position_matchup_volatility,
    s.best_matchup_differential,
    s.recent_matchup_performance
ORDER BY s.Season, s.Round, s.avg_pir_differential DESC