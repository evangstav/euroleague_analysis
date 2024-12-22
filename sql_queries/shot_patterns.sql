CREATE OR REPLACE VIEW shot_patterns AS
WITH game_shots AS (
    SELECT 
        Season,
        Gamecode,
        ID_PLAYER as Player_ID,
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
        
        -- Court positioning
        AVG(COORD_X) as avg_shot_x,
        AVG(COORD_Y) as avg_shot_y,
        STDDEV(COORD_X) as spread_shot_x,
        STDDEV(COORD_Y) as spread_shot_y

    FROM read_parquet('{data_dir}/raw/shot_data_{season}.parquet')
    GROUP BY Season, Gamecode, ID_PLAYER
),

shot_metrics AS (
    SELECT 
        gs.*,
    -- Shooting percentages
    CAST(made_shots AS FLOAT) / NULLIF(total_shots, 0) as fg_percentage,
    CAST(made_2pt AS FLOAT) / NULLIF(attempts_2pt, 0) as fg_percentage_2pt,
    CAST(made_3pt AS FLOAT) / NULLIF(attempts_3pt, 0) as fg_percentage_3pt,
    
    -- Special plays efficiency
    CAST(fastbreak_points AS FLOAT) / NULLIF(fastbreak_attempts, 0) as fastbreak_ppp,
    CAST(second_chance_points AS FLOAT) / NULLIF(second_chance_attempts, 0) as second_chance_ppp,
    
    -- Shot distribution ratios
    CAST(attempts_3pt AS FLOAT) / NULLIF(total_shots, 0) as three_point_rate,
    CAST(fastbreak_attempts AS FLOAT) / NULLIF(total_shots, 0) as fastbreak_rate,
    CAST(second_chance_attempts AS FLOAT) / NULLIF(total_shots, 0) as second_chance_rate,
    
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
    CAST(shots_zone_j AS FLOAT) / NULLIF(total_shots, 0) as zone_j_rate,
    
    -- Rolling averages (3 games)
    AVG(fg_percentage) OVER (
        PARTITION BY Player_ID 
        ORDER BY Season, Gamecode 
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) as fg_percentage_ma3,
    
    AVG(fastbreak_ppp) OVER (
        PARTITION BY Player_ID 
        ORDER BY Season, Gamecode 
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) as fastbreak_ppp_ma3

FROM game_shots gs
WHERE total_shots > 0
ORDER BY Season, Gamecode, total_shots DESC
)

SELECT * FROM shot_metrics
