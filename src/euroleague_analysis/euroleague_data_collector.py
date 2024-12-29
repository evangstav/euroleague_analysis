"""
Euroleague Data Collection Module with DVC pipeline support
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.play_by_play_data import PlayByPlay
from euroleague_api.shot_data import ShotData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> Dict[str, Any]:
    """Load parameters from params.yaml"""
    with open(params_path) as f:
        params = yaml.safe_load(f).get("extract", {})
    return params


def extract_box_score_data(season: int, data_dir: Path, competition_code: str = "E"):
    """Extract box score data for a given season and save to parquet."""
    logger.info(f"Extracting box score data for season {season}")

    client = BoxScoreData(competition_code)
    outputs = {}

    try:
        # Player stats
        player_stats = client.get_player_boxscore_stats_single_season(season)
        player_stats_path = data_dir / "raw" / f"player_stats_{season}.parquet"
        if not player_stats.empty:
            player_stats.to_parquet(player_stats_path)
            outputs["player_stats"] = str(player_stats_path)
            logger.info(f"Saved player stats for season {season}")

        # Quarter data
        quarter_data = client.get_game_boxscore_quarter_data_single_season(
            season, "ByQuarter"
        )
        quarter_data_path = data_dir / "raw" / f"quarter_data_{season}.parquet"
        if not quarter_data.empty:
            quarter_data.to_parquet(quarter_data_path)
            outputs["quarter_data"] = str(quarter_data_path)
            logger.info(f"Saved quarter data for season {season}")

        # End quarter data
        end_quarter_data = client.get_game_boxscore_quarter_data_single_season(
            season, "EndOfQuarter"
        )
        end_quarter_path = data_dir / "raw" / f"end_quarter_data_{season}.parquet"
        if not end_quarter_data.empty:
            end_quarter_data.to_parquet(end_quarter_path)
            outputs["end_quarter_data"] = str(end_quarter_path)
            logger.info(f"Saved end quarter data for season {season}")

        return outputs

    except Exception as e:
        logger.error(f"Error extracting box score data: {str(e)}")
        raise e


def extract_shot_data(season: int, data_dir: Path, competition_code: str = "E"):
    """Extract shot data for a given season and save to parquet."""
    logger.info(f"Extracting shot data for season {season}")

    client = ShotData(competition_code)
    try:
        shot_data = client.get_game_shot_data_single_season(season)
        shot_data_path = data_dir / "raw" / f"shot_data_{season}.parquet"
        if not shot_data.empty:
            shot_data.to_parquet(shot_data_path)
            logger.info(f"Saved shot data for season {season}")
            return {"shot_data": str(shot_data_path)}
        return {}
    except Exception as e:
        logger.error(f"Error extracting shot data: {str(e)}")
        raise e


def extract_playbyplay_data(season: int, data_dir: Path, competition_code: str = "E"):
    """Extract play-by-play data for a given season and save to parquet."""
    logger.info(f"Extracting play-by-play data for season {season}")

    client = PlayByPlay(competition_code)
    try:
        pbp_data = client.get_game_play_by_play_data_single_season(season)
        pbp_path = data_dir / "raw" / f"playbyplay_data_{season}.parquet"
        if not pbp_data.empty:
            pbp_data.to_parquet(pbp_path)
            logger.info(f"Saved play-by-play data for season {season}")
            return {"playbyplay_data": str(pbp_path)}
        return {}
    except Exception as e:
        logger.error(f"Error extracting play-by-play data: {str(e)}")
        raise e


def setup_directories(data_dir: str):
    """Create necessary data directories."""
    data_path = Path(data_dir)
    (data_path / "raw").mkdir(parents=True, exist_ok=True)
    return data_path


def main():
    # Load parameters
    params = load_params()
    season = params.get("season", 2023)
    data_dir = params.get("data_dir", "euroleague_data")
    competition_code = params.get("competition_code", "E")

    # Setup directories
    data_path = setup_directories(data_dir)

    # Extract all data types
    outputs = {}
    outputs.update(extract_box_score_data(season, data_path, competition_code))
    outputs.update(extract_shot_data(season, data_path, competition_code))
    outputs.update(extract_playbyplay_data(season, data_path, competition_code))

    # Save outputs for DVC
    with open("extract_outputs.json", "w") as f:
        json.dump(outputs, f)


if __name__ == "__main__":
    main()
