from pathlib import Path

import polars as pl

data_dir = Path("euroleague_data")

try:
    # Try direct path
    player_stats_path = data_dir / "2023" / "player_stats.parquet"
    if not player_stats_path.exists():
        # Try raw subdirectory
        player_stats_path = data_dir / "raw" / "player_stats_2023.parquet"

    player_stats = pl.read_parquet(str(player_stats_path))
    print("\nPlayer Stats Columns:")
    print(player_stats.columns)

    # Try direct path
    playbyplay_path = data_dir / "2023" / "playbyplay_data.parquet"
    if not playbyplay_path.exists():
        # Try raw subdirectory
        playbyplay_path = data_dir / "raw" / "playbyplay_data_2023.parquet"

    playbyplay = pl.read_parquet(str(playbyplay_path))
    print("\nPlaybyplay Columns:")
    print(playbyplay.columns)

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("\nTrying to list available files in data directory:")
    if data_dir.exists():
        print("\nFiles in data_dir:")
        for p in data_dir.rglob("*.parquet"):
            print(f"  {p}")
    else:
        print(f"\nData directory {data_dir} does not exist")
