CREATE OR REPLACE VIEW player_stats_features AS
WITH player_stats AS (
    SELECT 
        *,
        convert_minutes(Minutes) as minutes_float
    FROM read_parquet('{data_dir}/raw/player_stats_{season}.parquet')
),
-- Rest of the rolling_stats SQL query...
