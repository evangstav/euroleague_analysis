"""
Euroleague Data Collection Module for Player Performance Analysis
This module collects and structures data for analyzing player PIR predictions.
"""

from typing import List
import pandas as pd
import numpy as np
import logging
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.shot_data import ShotData
from euroleague_api.play_by_play_data import PlayByPlay

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EuroleagueDataCollector:
    def __init__(self, competition_code: str = "E"):
        """
        Initialize the data collector

        Args:
            competition_code: Competition code (default "E" for Euroleague)
        """
        self.competition_code = competition_code
        self.boxscore_client = BoxScoreData(competition_code)
        self.shot_data_client = ShotData(competition_code)
        self.play_by_play_client = PlayByPlay(competition_code)

    def collect_player_stats(self, season: int) -> pd.DataFrame:
        """
        Collect detailed player statistics for a season

        Args:
            season: The season to collect data for (e.g., 2023)

        Returns:
            DataFrame containing player statistics including PIR
        """
        logger.info(f"Collecting player statistics for season {season}")

        try:
            # Get player boxscore stats for the season
            player_stats = self.boxscore_client.get_player_boxscore_stats_single_season(
                season
            )

            # Add shot data if available
            try:
                shot_data = self.shot_data_client.get_game_shot_data_single_season(
                    season
                )
                if not shot_data.empty:
                    # Aggregate shot data by player and game
                    shot_metrics = self._process_shot_data(shot_data)
                    # Merge with player stats
                    player_stats = player_stats.merge(
                        shot_metrics, on=["Season", "Gamecode", "Player_ID"], how="left"
                    )
            except Exception as e:
                logger.warning(f"Error collecting shot data: {str(e)}")

            return player_stats

        except Exception as e:
            logger.error(f"Error collecting player statistics: {str(e)}")
            return pd.DataFrame()

    def _process_shot_data(self, shot_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw shot data to create player shooting metrics

        Args:
            shot_data: DataFrame containing raw shot data

        Returns:
            DataFrame with aggregated shooting metrics by player and game
        """
        # Group by game and player
        shot_metrics = (
            shot_data.groupby(["Season", "Gamecode", "Player_ID"])
            .agg(
                {
                    "shot_made": ["sum", "count"],
                    "shot_zone": lambda x: x.value_counts().to_dict(),
                    "shot_type": lambda x: x.value_counts().to_dict(),
                }
            )
            .reset_index()
        )

        # Flatten column names
        shot_metrics.columns = [
            "Season",
            "Gamecode",
            "Player_ID",
            "shots_made",
            "shots_attempted",
            "shot_zones",
            "shot_types",
        ]

        return shot_metrics

    def collect_team_performance(self, season: int) -> pd.DataFrame:
        """
        Collect team performance data including opponent stats

        Args:
            season: Season to collect data for

        Returns:
            DataFrame with team performance metrics
        """
        logger.info(f"Collecting team performance data for season {season}")

        try:
            # Get quarter by quarter data for all games
            quarters_data = (
                self.boxscore_client.get_game_boxscore_quarter_data_single_season(
                    season, boxscore_type="ByQuarter"
                )
            )

            # Get end of quarter data for pace and style metrics
            end_quarter_data = (
                self.boxscore_client.get_game_boxscore_quarter_data_single_season(
                    season, boxscore_type="EndOfQuarter"
                )
            )

            # Merge quarter and end of quarter data
            if not quarters_data.empty and not end_quarter_data.empty:
                team_data = quarters_data.merge(
                    end_quarter_data,
                    on=["Season", "Gamecode"],
                    how="left",
                    suffixes=("", "_end"),
                )
                return team_data

            return quarters_data

        except Exception as e:
            logger.error(f"Error collecting team performance data: {str(e)}")
            return pd.DataFrame()

    def prepare_pir_prediction_dataset(
        self, seasons: List[int], lookback_games: int = 5
    ) -> pd.DataFrame:
        """
        Prepare complete dataset for PIR prediction modeling

        Args:
            seasons: List of seasons to include
            lookback_games: Number of previous games to use for rolling stats

        Returns:
            DataFrame ready for PIR prediction modeling
        """
        all_data = []

        for season in seasons:
            # Collect all data sources
            player_stats = self.collect_player_stats(season)
            team_performance = self.collect_team_performance(season)

            if player_stats.empty:
                logger.warning(f"No player stats available for season {season}")
                continue

            # Process player statistics
            if not player_stats.empty:
                # Sort by player and date
                player_stats = player_stats.sort_values(
                    ["Player_ID", "Season", "Gamecode"]
                )

                # Calculate rolling averages for key metrics
                metrics = ["PIR", "Minutes", "Points", "Ast", "RebD", "RebO"]
                # TODO
                # convert all metrics  to numeric
                # only select the ones that are possible to aggregate
                for metric in metrics:
                    if metric in player_stats.columns:
                        player_stats[f"{metric}_rolling_avg"] = player_stats.groupby(
                            "Player_ID"
                        )[metric].transform(
                            lambda x: x.rolling(lookback_games, min_periods=1).mean()
                        )

                # Add team context if available
                if not team_performance.empty:
                    player_stats = player_stats.merge(
                        team_performance, on=["Season", "Gamecode"], how="left"
                    )

                all_data.append(player_stats)

        if not all_data:
            logger.error("No data collected for any season")
            return pd.DataFrame()

        # Combine all seasons
        final_dataset = pd.concat(all_data, ignore_index=True)

        # Handle missing values
        numeric_cols = final_dataset.select_dtypes(include=[np.number]).columns
        final_dataset[numeric_cols] = final_dataset[numeric_cols].fillna(0)

        return final_dataset


if __name__ == "__main__":
    # Initialize collector
    collector = EuroleagueDataCollector()

    # Collect data for multiple seasons
    dataset = collector.prepare_pir_prediction_dataset(seasons=[2023], lookback_games=5)

    # Save to CSV
    if not dataset.empty:
        dataset.to_csv("euroleague_pir_prediction_data.csv", index=False)
        print(f"Dataset saved with {len(dataset)} records")
    else:
        print("No data collected")
