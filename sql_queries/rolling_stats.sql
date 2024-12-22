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

-- Add rolling averages and game-by-game rankings
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
    END as improving_form,
    
    -- Consistency metrics
    STDDEV(PIR) OVER (
        PARTITION BY Player_ID 
        ORDER BY Season, Round 
        ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
    ) as pir_std3

FROM game_stats
WHERE minutes_played > 0  -- Filter out DNPs
ORDER BY Season, Round, Gamecode, PIR DESC
