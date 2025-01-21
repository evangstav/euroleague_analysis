CREATE OR REPLACE VIEW player_stats_features_{season} AS
WITH player_stats AS (
    SELECT 
        *,
        convert_minutes(Minutes) as minutes_float
    FROM read_parquet('{data_dir}/raw/player_stats_{season}.parquet', HIVE_PARTITIONING=1)
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
    WHERE convert_minutes(Minutes) > 0
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
    FROM read_parquet('{data_dir}/raw/playbyplay_data_{season}.parquet', HIVE_PARTITIONING=1)
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
        
    FROM read_parquet('{data_dir}/raw/playbyplay_data_{season}.parquet', HIVE_PARTITIONING=1) p
    LEFT JOIN consecutive_plays c USING(PLAYER_ID, Gamecode)
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
    FROM read_parquet('{data_dir}/raw/playbyplay_data_{season}.parquet', HIVE_PARTITIONING=1)
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
    FROM read_parquet('{data_dir}/raw/shot_data_{season}.parquet', HIVE_PARTITIONING=1) s
    LEFT JOIN read_parquet('{data_dir}/raw/playbyplay_data_{season}.parquet', HIVE_PARTITIONING=1) p 
        USING(ID_PLAYER, Gamecode)
    GROUP BY s.ID_PLAYER, s.Gamecode
),

player_features AS (
    SELECT 
        g.*,
        f.clutch_plays,
        f.clutch_scores,
        f.first_quarter_plays,
        f.fourth_quarter_plays,
        f.close_game_plays,
        f.unique_play_types,
        f.consecutive_positive_plays,
        u.assist_rate,
        u.shot_attempt_rate,
        u.defensive_play_rate,
        u.turnover_rate,
        c.shot_quality_index,
        c.court_coverage_area,
        c.unique_scoring_zones,
        c.defensive_versatility,
        
        -- Rolling averages (3 games)
        AVG(g.minutes_played) OVER player_window as minutes_ma3,
        AVG(g.Points) OVER player_window as points_ma3,
        AVG(g.PIR) OVER player_window as pir_ma3,
        AVG(g.fg_percentage) OVER player_window as fg_percentage_ma3,
        AVG(c.shot_quality_index) OVER player_window as shot_quality_ma3,
        AVG(f.clutch_scores) OVER player_window as clutch_scoring_ma3,
        
        -- Game rankings
        RANK() OVER game_window as pir_rank_in_game,
        g.PIR - AVG(g.PIR) OVER (PARTITION BY g.Player_ID, g.Season) as pir_vs_season_avg,
        CASE 
            WHEN g.PIR > LAG(g.PIR) OVER (PARTITION BY g.Player_ID ORDER BY g.Season, g.Round) THEN 1
            ELSE 0
        END as improving_form,
        STDDEV(g.PIR) OVER player_window as pir_std3

    FROM game_stats g
    LEFT JOIN (
        SELECT * FROM game_flow 
        JOIN player_usage USING(PLAYER_ID, Gamecode)
        JOIN performance_context USING(PLAYER_ID, Gamecode)
    ) combined USING(PLAYER_ID, Gamecode)
    WINDOW 
        player_window AS (
            PARTITION BY Player_ID 
            ORDER BY Season, Round 
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ),
        game_window AS (
            PARTITION BY Gamecode 
            ORDER BY PIR DESC
        )
)

SELECT * FROM player_features
ORDER BY Season, Round, Gamecode, PIR DESC