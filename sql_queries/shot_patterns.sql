CREATE OR REPLACE VIEW shot_patterns AS
WITH game_shots AS (
    SELECT 
        Season,
        Gamecode,
        ID_PLAYER as Player_ID,
        -- Rest of the shot_patterns SQL query...
