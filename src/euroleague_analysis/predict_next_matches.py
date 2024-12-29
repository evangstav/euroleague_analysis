"""
Module for generating features for next Euroleague matches predictions.
Leverages the existing feature database and views.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_features_df(
    conn: duckdb.DuckDBPyConnection,
    team: str,
    data_dir: Path,
    season: int,
    last_n: int = 5,
) -> pd.DataFrame:
    """Load recent team features from parquet file."""

    query = f"""
        WITH team_features AS (
            SELECT 
                Player_ID,
                PlayerName,
                Team,
                Season,
                Round,
                minutes_played,
                PIR,
                Points,
                fg_percentage,
                fg_percentage_2pt,
                ft_percentage,
                ast_to_turnover,
                rebounds_per_minute,
                defensive_plays_per_minute,
                is_starter,
                minutes_ma3,
                points_ma3,
                pir_ma3,
                -- Additional metrics
                ROW_NUMBER() OVER (PARTITION BY Player_ID ORDER BY Season DESC, Round DESC) as recency
            FROM read_parquet('{data_dir}/features.parquet')
            WHERE Team = '{team}'
              AND Season = {season}
            QUALIFY recency <= {last_n}
        )
        SELECT 
            Team,
            PlayerName,
            AVG(minutes_played) as avg_minutes,
            AVG(PIR) as avg_pir,
            AVG(Points) as avg_points,
            AVG(fg_percentage) as avg_fg_pct,
            AVG(ft_percentage) as avg_ft_pct,
            AVG(ast_to_turnover) as avg_ast_to,
            AVG(pir_ma3) as pir_trend,
            AVG(points_ma3) as points_trend,
            COUNT(*) as games_played,
            MAX(is_starter::INTEGER) as is_starter
        FROM team_features
        GROUP BY Team, PlayerName
        HAVING COUNT(*) >= 3
        ORDER BY avg_pir DESC
    """

    try:
        return conn.execute(query).df()
    except Exception as e:
        logger.error(f"Error loading features for {team}: {str(e)}")
        return pd.DataFrame()


def generate_matchup_features(
    matchups: List[Tuple[str, str]],
    data_dir: Path = Path("euroleague_data"),
    season: int = 2024,
) -> Dict[str, Any]:
    """Generate features for upcoming matchups."""

    # Initialize DB connection
    conn = duckdb.connect()
    features = {}

    for home_team, away_team in matchups:
        logger.info(f"Processing {home_team} vs {away_team}")

        try:
            # Get player stats
            home_players = load_features_df(conn, home_team, data_dir, season)
            away_players = load_features_df(conn, away_team, data_dir, season)

            # Calculate team aggregates
            home_team_stats = {
                "avg_team_pir": home_players["avg_pir"].mean(),
                "avg_team_points": home_players["avg_points"].mean(),
                "rotation_size": len(home_players),
                "starters_pir": home_players[home_players["is_starter"] == 1][
                    "avg_pir"
                ].mean(),
                "bench_pir": home_players[home_players["is_starter"] == 0][
                    "avg_pir"
                ].mean(),
            }

            away_team_stats = {
                "avg_team_pir": away_players["avg_pir"].mean(),
                "avg_team_points": away_players["avg_points"].mean(),
                "rotation_size": len(away_players),
                "starters_pir": away_players[away_players["is_starter"] == 1][
                    "avg_pir"
                ].mean(),
                "bench_pir": away_players[away_players["is_starter"] == 0][
                    "avg_pir"
                ].mean(),
            }

            features[f"{home_team}_vs_{away_team}"] = {
                "team_stats": {
                    "home_team": home_team_stats,
                    "away_team": away_team_stats,
                },
                "player_stats": {
                    "home_team": home_players.to_dict("records"),
                    "away_team": away_players.to_dict("records"),
                },
            }

        except Exception as e:
            logger.error(f"Error processing matchup: {str(e)}")
            features[f"{home_team}_vs_{away_team}"] = {"error": str(e)}

    conn.close()
    return features


if __name__ == "__main__":
    # Example usage
    matchups = [("Real Madrid", "Barcelona"), ("Olympiacos", "Panathinaikos")]

    features = generate_matchup_features(matchups)
    print("\nMatchup Features:")
    print("================")

    for matchup, data in features.items():
        print(f"\n{matchup}")
        if "error" in data:
            print(f"Error: {data['error']}")
            continue

        home_team = matchup.split("_vs_")[0]
        away_team = matchup.split("_vs_")[1]

        print(f"\n{home_team} Team Stats:")
        print(pd.DataFrame([data["team_stats"]["home_team"]]).to_string())

        print(f"\n{away_team} Team Stats:")
        print(pd.DataFrame([data["team_stats"]["away_team"]]).to_string())

        print(f"\n{home_team} Top Players:")
        home_players = pd.DataFrame(data["player_stats"]["home_team"])
        print(
            home_players[["PlayerName", "avg_pir", "avg_points", "games_played"]]
            .head()
            .to_string()
        )

        print(f"\n{away_team} Top Players:")
        away_players = pd.DataFrame(data["player_stats"]["away_team"])
        print(
            away_players[["PlayerName", "avg_pir", "avg_points", "games_played"]]
            .head()
            .to_string()
        )
