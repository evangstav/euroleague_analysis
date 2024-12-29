CREATE OR REPLACE VIEW player_stats_features_{season} AS
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
        CAST(Points as FLOAT) / NULLIF(minutes_played, 0) as points_per_minute,
        CAST(FieldGoalsMade2 + FieldGoalsMade3 as FLOAT) / NULLIF(FieldGoalsAttempted2 + FieldGoalsAttempted3, 0) as fg_percentage,
        CAST(FieldGoalsMade2 as FLOAT) / NULLIF(FieldGoalsAttempted2, 0) as fg_percentage_2pt,
        CAST(FieldGoalsMade3 as FLOAT) / NULLIF(FieldGoalsAttempted3, 0) as fg_percentage_3pt,
        CAST(FreeThrowsMade as FLOAT) / NULLIF(FreeThrowsAttempted, 0) as ft_percentage,
        
        -- Shot distribution
        CAST(FieldGoalsAttempted3 as FLOAT) / NULLIF(FieldGoalsAttempted2 + FieldGoalsAttempted3, 0) as three_point_attempt_rate,
        
        -- Advanced metrics
        CAST(Assistances as FLOAT) / NULLIF(Turnovers, 0) as ast_to_turnover,
        CAST(OffensiveRebounds + DefensiveRebounds as FLOAT) / NULLIF(minutes_played, 0) as rebounds_per_minute,
        CAST(Steals + BlocksFavour as FLOAT) / NULLIF(minutes_played, 0) as defensive_plays_per_minute
        
    FROM player_stats
),

consecutive_plays AS (
    SELECT 
        PLAYER_ID,
        Gamecode,
        NUMBEROFPLAY,
        CASE WHEN PLAYTYPE IN ('2FGM', '3FGM', 'FTM', 'D', 'O', 'AS') THEN 1 ELSE 0 END as is_positive_play,
        CASE WHEN 
            PLAYTYPE IN ('2FGM', '3FGM', 'FTM', 'D', 'O', 'AS') AND
            LEAD(PLAYTYPE) OVER (PARTITION BY PLAYER_ID, Gamecode ORDER BY NUMBEROFPLAY) IN ('2FGM', '3FGM', 'FTM', 'D', 'O', 'AS')
        THEN 1 ELSE 0 END as is_consecutive
    FROM read_parquet('{data_dir}/raw/playbyplay_data_{season}.parquet')
),

game_flow AS (
    SELECT 
        p.PLAYER_ID,
        p.Gamecode,
        -- Clutch performance (last 5 minutes of game)
        COUNT(CASE WHEN p.PERIOD = 4 AND p.MINUTE >= 35 THEN 1 END) as clutch_plays,
        COUNT(CASE WHEN p.PERIOD = 4 AND p.MINUTE >= 35 AND p.PLAYTYPE LIKE '%Made%' THEN 1 END) as clutch_scores,
        
        -- Period-by-period activity
        COUNT(CASE WHEN p.PERIOD = 1 THEN 1 END) as first_quarter_plays,
        COUNT(CASE WHEN p.PERIOD = 4 THEN 1 END) as fourth_quarter_plays,
        
        -- Game situation impact
        COUNT(CASE WHEN ABS(CAST(p.POINTS_A as FLOAT) - CAST(p.POINTS_B as FLOAT)) <= 5 THEN 1 END) as close_game_plays,
        
        -- Play type diversity
        COUNT(DISTINCT p.PLAYTYPE) as unique_play_types,
        
        -- Momentum indicators
        SUM(c.is_consecutive) as consecutive_positive_plays
        
    FROM read_parquet('{data_dir}/raw/playbyplay_data_{season}.parquet') p
    LEFT JOIN consecutive_plays c ON p.PLAYER_ID = c.PLAYER_ID AND p.Gamecode = c.Gamecode
    GROUP BY p.PLAYER_ID, p.Gamecode
),

player_usage AS (
    SELECT 
        PLAYER_ID,
        Gamecode,
        -- Play initiation
        CAST(COUNT(CASE WHEN PLAYTYPE = 'AS' THEN 1 END) AS FLOAT) * 100.0 / 
            NULLIF(COUNT(*), 0) as assist_rate,
        
        -- Scoring responsibility
        CAST(COUNT(CASE WHEN PLAYTYPE IN ('2FGA', '3FGA') THEN 1 END) AS FLOAT) * 100.0 / 
            NULLIF(COUNT(*), 0) as shot_attempt_rate,
        
        -- Defensive involvement
        CAST(COUNT(CASE WHEN PLAYTYPE IN ('D', 'BLK', 'ST') THEN 1 END) AS FLOAT) * 100.0 / 
            NULLIF(COUNT(*), 0) as defensive_play_rate,
        
        -- Ball handling
        CAST(COUNT(CASE WHEN PLAYTYPE = 'TO' THEN 1 END) AS FLOAT) * 100.0 / 
            NULLIF(COUNT(CASE WHEN PLAYTYPE IN ('2FGA', '3FGA', 'AS') THEN 1 END), 0) as turnover_rate
    FROM read_parquet('{data_dir}/raw/playbyplay_data_{season}.parquet')
    GROUP BY PLAYER_ID, Gamecode
),

performance_context AS (
    SELECT 
        s.ID_PLAYER as PLAYER_ID,
        s.Gamecode,
        -- Shot quality metrics
        AVG(CASE 
            WHEN s.FASTBREAK = 'Yes' THEN 1.2
            WHEN s.SECOND_CHANCE = 'Yes' THEN 1.1
            ELSE 1.0
        END) as shot_quality_index,
        
        -- Court coverage
        STDDEV(s.COORD_X) * STDDEV(s.COORD_Y) as court_coverage_area,
        
        -- Scoring versatility
        COUNT(DISTINCT s.ZONE) as unique_scoring_zones,
        
        -- Defensive positioning (from playbyplay)
        COUNT(DISTINCT p.PLAYTYPE) FILTER (WHERE p.PLAYTYPE IN ('BLK', 'ST', 'D')) as defensive_versatility
    FROM read_parquet('{data_dir}/raw/shot_data_{season}.parquet') s
    LEFT JOIN read_parquet('{data_dir}/raw/playbyplay_data_{season}.parquet') p 
        ON s.ID_PLAYER = p.PLAYER_ID AND s.Gamecode = p.Gamecode
    GROUP BY s.ID_PLAYER, s.Gamecode
),

player_features AS (
    SELECT 
        g.*,
        -- Game flow features
        f.clutch_plays,
        f.clutch_scores,
        f.first_quarter_plays,
        f.fourth_quarter_plays,
        f.close_game_plays,
        f.unique_play_types,
        f.consecutive_positive_plays,
        
        -- Usage patterns
        u.assist_rate,
        u.shot_attempt_rate,
        u.defensive_play_rate,
        u.turnover_rate,
        
        -- Performance context
        c.shot_quality_index,
        c.court_coverage_area,
        c.unique_scoring_zones,
        c.defensive_versatility,
        
        -- Rolling averages (3 games)
        AVG(g.minutes_played) OVER (
            PARTITION BY g.Player_ID 
            ORDER BY g.Season, g.Round 
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as minutes_ma3,
        
        AVG(g.Points) OVER (
            PARTITION BY g.Player_ID 
            ORDER BY g.Season, g.Round 
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as points_ma3,
        
        AVG(g.PIR) OVER (
            PARTITION BY g.Player_ID 
            ORDER BY g.Season, g.Round 
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as pir_ma3,
        
        AVG(g.fg_percentage) OVER (
            PARTITION BY g.Player_ID 
            ORDER BY g.Season, g.Round 
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as fg_percentage_ma3,
        
        -- New rolling averages for added features
        AVG(c.shot_quality_index) OVER (
            PARTITION BY g.Player_ID 
            ORDER BY g.Season, g.Round 
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as shot_quality_ma3,
        
        AVG(f.clutch_scores) OVER (
            PARTITION BY g.Player_ID 
            ORDER BY g.Season, g.Round 
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as clutch_scoring_ma3,
        
        -- Game-by-game rankings
        RANK() OVER (
            PARTITION BY g.Gamecode 
            ORDER BY g.PIR DESC
        ) as pir_rank_in_game,
        
        -- Difference from season average
        g.PIR - AVG(g.PIR) OVER (PARTITION BY g.Player_ID, g.Season) as pir_vs_season_avg,
        
        -- Form indicators
        CASE 
            WHEN g.PIR > LAG(g.PIR) OVER (PARTITION BY g.Player_ID ORDER BY g.Season, g.Round) THEN 1
            ELSE 0
        END as improving_form,
        
        -- Consistency metrics
        STDDEV(g.PIR) OVER (
            PARTITION BY g.Player_ID 
            ORDER BY g.Season, g.Round 
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) as pir_std3

    FROM game_stats g
    LEFT JOIN game_flow f ON g.Player_ID = f.PLAYER_ID AND g.Gamecode = f.Gamecode
    LEFT JOIN player_usage u ON g.Player_ID = u.PLAYER_ID AND g.Gamecode = u.Gamecode
    LEFT JOIN performance_context c ON g.Player_ID = c.PLAYER_ID AND g.Gamecode = c.Gamecode
    WHERE g.minutes_played > 0  -- Filter out DNPs
)

SELECT * FROM player_features
ORDER BY Season, Round, Gamecode, PIR DESC
