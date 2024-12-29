"""
Module for mapping teams and players for a given season.
"""

import logging
import yaml
from pathlib import Path
import pandas as pd
from typing import Dict, Any
from euroleague_api.player_stats import PlayerStats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load parameters from params.yaml"""
    with open(params_path) as f:
        params = yaml.safe_load(f).get("extract", {})
    return params


def collect_season_mappings(season: int, data_dir: Path, competition_code: str = "E"):
    """
    Collect team and player mappings for a given season.
    """
    logger.info(f"Collecting team-player mappings for season {season}")

    # Initialize client
    client = PlayerStats(competition_code)

    try:
        # Get traditional stats - this includes team and player info
        df = client.get_player_stats_single_season(
            endpoint="traditional",
            season=season,
            phase_type_code=None,  # Get all phases
            statistic_mode="Accumulated",  # Get accumulated stats to ensure all players
        )

        if df.empty:
            logger.warning(f"No data found for season {season}")
            return None

        # Extract key mapping fields
        mapping_df = df[
            [
                "playerId",
                "playerName",
                "gamesPlayed",
                "minutes",
                "teamCode",
                "teamName",
                "playerDorsal",
                "playerFlag",
                "playerHeight",
                "playerPosition",
            ]
        ].copy()

        # Rename columns for consistency
        mapping_df.columns = [
            "player_id",
            "player_name",
            "games_played",
            "minutes",
            "team_code",
            "team_name",
            "jersey_number",
            "nationality",
            "height",
            "position",
        ]

        # Add season
        mapping_df["season"] = season

        # Clean up height data
        mapping_df["height"] = pd.to_numeric(
            mapping_df["height"].str.replace(",", "."), errors="coerce"
        )

        # Add basic stats thresholds
        mapping_df["is_regular"] = mapping_df["games_played"] >= 10
        mapping_df["minutes_per_game"] = (
            pd.to_numeric(mapping_df["minutes"].str.split(":").str[0], errors="coerce")
            + pd.to_numeric(
                mapping_df["minutes"].str.split(":").str[1], errors="coerce"
            )
            / 60
        ) / mapping_df["games_played"]

        # Create team info DataFrame
        team_info = (
            mapping_df[["team_code", "team_name"]]
            .drop_duplicates()
            .assign(season=season)
        )

        # Save to parquet
        output_dir = data_dir / "mappings"
        output_dir.mkdir(exist_ok=True, parents=True)

        # Save mappings
        mapping_file = output_dir / f"player_team_mapping_{season}.parquet"
        mapping_df.to_parquet(mapping_file)
        logger.info(f"Saved player-team mappings to {mapping_file}")

        # Save team info
        team_file = output_dir / f"team_info_{season}.parquet"
        team_info.to_parquet(team_file)
        logger.info(f"Saved team info to {team_file}")

        return {
            "mapping_file": str(mapping_file),
            "team_file": str(team_file),
            "n_players": len(mapping_df),
            "n_teams": len(team_info),
        }

    except Exception as e:
        logger.error(f"Error collecting mappings: {str(e)}")
        raise e


def main():
    # Load parameters
    params = load_params()
    season = params.get("season", 2024)
    data_dir = Path(params.get("data_dir", "euroleague_data"))
    competition_code = params.get("competition_code", "E")

    # Collect mappings
    outputs = collect_season_mappings(season, data_dir, competition_code)

    if outputs:
        # Print summary
        print("\nCollection Summary:")
        print(f"Total players: {outputs['n_players']}")
        print(f"Total teams: {outputs['n_teams']}")
        print(f"\nFiles saved:")
        print(f"Mappings: {outputs['mapping_file']}")
        print(f"Team info: {outputs['team_file']}")


if __name__ == "__main__":
    main()
