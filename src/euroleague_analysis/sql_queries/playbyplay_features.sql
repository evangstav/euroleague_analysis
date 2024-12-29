CREATE OR REPLACE VIEW playbyplay_features_{season} AS
WITH consecutive_plays AS (
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
)

SELECT 
    g.*,
    u.assist_rate,
    u.shot_attempt_rate,
    u.defensive_play_rate,
    u.turnover_rate
FROM game_flow g
JOIN player_usage u 
    ON g.PLAYER_ID = u.PLAYER_ID 
    AND g.Gamecode = u.Gamecode