CREATE OR REPLACE VIEW shot_patterns_{season} AS
WITH game_shots AS (
    SELECT
        season,
        gamecode,
        id_player AS player_id,
        -- Basic counting
        COUNT(*) AS total_shots,
        SUM(points) AS total_points,
        COUNT(*) FILTER (WHERE points > 0) AS made_shots,

        -- Shot types
        COUNT(*) FILTER (WHERE points = 2) AS attempts_2pt,
        COUNT(*) FILTER (WHERE points = 2 AND points > 0) AS made_2pt,
        COUNT(*) FILTER (WHERE points = 3) AS attempts_3pt,
        COUNT(*) FILTER (WHERE points = 3 AND points > 0) AS made_3pt,

        -- Special play types
        COUNT(*) FILTER (WHERE fastbreak = 1) AS fastbreak_attempts,
        SUM(points) FILTER (WHERE fastbreak = 1) AS fastbreak_points,
        COUNT(*) FILTER (WHERE second_chance = 1) AS second_chance_attempts,
        SUM(points) FILTER (WHERE second_chance = 1) AS second_chance_points,

        -- Court zones
        COUNT(*) FILTER (WHERE zone = 'A') AS shots_zone_a,
        COUNT(*) FILTER (WHERE zone = 'B') AS shots_zone_b,
        COUNT(*) FILTER (WHERE zone = 'C') AS shots_zone_c,
        COUNT(*) FILTER (WHERE zone = 'D') AS shots_zone_d,
        COUNT(*) FILTER (WHERE zone = 'E') AS shots_zone_e,
        COUNT(*) FILTER (WHERE zone = 'F') AS shots_zone_f,
        COUNT(*) FILTER (WHERE zone = 'G') AS shots_zone_g,
        COUNT(*) FILTER (WHERE zone = 'H') AS shots_zone_h,
        COUNT(*) FILTER (WHERE zone = 'I') AS shots_zone_i,
        COUNT(*) FILTER (WHERE zone = 'J') AS shots_zone_j,

        -- Court positioning
        AVG(coord_x) AS avg_shot_x,
        AVG(coord_y) AS avg_shot_y,
        STDDEV(coord_x) AS spread_shot_x,
        STDDEV(coord_y) AS spread_shot_y

    FROM READ_PARQUET('{data_dir}/raw/shot_data_{season}.parquet')
    GROUP BY season, gamecode, id_player
),

shot_metrics AS (
    SELECT
        gs.*,
        -- Shooting percentages
        CAST(made_shots AS FLOAT) / NULLIF(total_shots, 0) AS fg_percentage,
        CAST(made_2pt AS FLOAT) / NULLIF(attempts_2pt, 0) AS fg_percentage_2pt,
        CAST(made_3pt AS FLOAT) / NULLIF(attempts_3pt, 0) AS fg_percentage_3pt,

        -- Special plays efficiency
        CAST(fastbreak_points AS FLOAT)
        / NULLIF(fastbreak_attempts, 0) AS fastbreak_ppp,
        CAST(second_chance_points AS FLOAT)
        / NULLIF(second_chance_attempts, 0) AS second_chance_ppp,

        -- Shot distribution ratios
        CAST(attempts_3pt AS FLOAT)
        / NULLIF(total_shots, 0) AS three_point_rate,
        CAST(fastbreak_attempts AS FLOAT)
        / NULLIF(total_shots, 0) AS fastbreak_rate,
        CAST(second_chance_attempts AS FLOAT)
        / NULLIF(total_shots, 0) AS second_chance_rate,

        -- Zone preferences (normalized)
        CAST(shots_zone_a AS FLOAT) / NULLIF(total_shots, 0) AS zone_a_rate,
        CAST(shots_zone_b AS FLOAT) / NULLIF(total_shots, 0) AS zone_b_rate,
        CAST(shots_zone_c AS FLOAT) / NULLIF(total_shots, 0) AS zone_c_rate,
        CAST(shots_zone_d AS FLOAT) / NULLIF(total_shots, 0) AS zone_d_rate,
        CAST(shots_zone_e AS FLOAT) / NULLIF(total_shots, 0) AS zone_e_rate,
        CAST(shots_zone_f AS FLOAT) / NULLIF(total_shots, 0) AS zone_f_rate,
        CAST(shots_zone_g AS FLOAT) / NULLIF(total_shots, 0) AS zone_g_rate,
        CAST(shots_zone_h AS FLOAT) / NULLIF(total_shots, 0) AS zone_h_rate,
        CAST(shots_zone_i AS FLOAT) / NULLIF(total_shots, 0) AS zone_i_rate,
        CAST(shots_zone_j AS FLOAT) / NULLIF(total_shots, 0) AS zone_j_rate,

        -- Rolling averages (3 games)
        AVG(fg_percentage) OVER (
            PARTITION BY player_id
            ORDER BY season, gamecode
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS fg_percentage_ma3,

        AVG(fastbreak_ppp) OVER (
            PARTITION BY player_id
            ORDER BY season, gamecode
            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS fastbreak_ppp_ma3

    FROM game_shots gs
    WHERE total_shots > 0
    ORDER BY season, gamecode, total_shots DESC
)

SELECT * FROM shot_metrics
