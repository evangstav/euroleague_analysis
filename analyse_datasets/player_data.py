import duckdb
import pandas as pd
from functools import reduce

duckdb.connect()

dset = duckdb.sql(
    """--sql
WITH parsed_data AS (
    SELECT 
        *,
        COALESCE(
            strptime(Minutes, '%M:%S')::TIME,
            TIME '00:00:00'
        ) AS parsed_minutes
    FROM read_parquet('euroleague_data/raw/player_stats_2023.parquet')
    WHERE
        Minutes IS NULL
        OR REGEXP_MATCHES(Minutes, '^\d{2}:\d{2}$')
)
SELECT
    *, 
    EXTRACT(MINUTE FROM parsed_minutes) * 60 + EXTRACT(SECOND FROM parsed_minutes) AS total_seconds
FROM parsed_data;
    """
)


dset.columns
dset.describe()


dset.show()

dset["Minutes"]

duckdb.sql("""---sql
            SELECT 
                Player_ID,
                Gamecode,
                AVG(Points) OVER (
                    PARTITION BY Player_ID
                    ORDER BY Gamecode
                    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
                    ) as avg_points_last_5_games
            FROM dset
            WHERE HOME=0
           """)


def generate_moving_stats_query(
    table_name: str,
    column_name: str,
    window_sizes: list[int],
    agg_function: str = "AVG",
    include_current: bool = True,
) -> str:
    """
    Generate a DuckDB query for calculating moving statistics with home/away splits.

    Parameters:
    -----------
    table_name : str
        Name of the table containing the data
    column_name : str
        Name of the column to calculate statistics for (e.g., 'points', 'rebounds')
    window_sizes : list[int]
        List of window sizes for moving calculations
    agg_function : str, default 'AVG'
        Aggregation function to use ('AVG', 'MEDIAN', 'SUM', etc.)
    include_current : bool, default True
        Whether to include current game in the window calculation

    Returns:
    --------
    str
        DuckDB SQL query string

    Example:
    --------
    >>> query = generate_moving_stats_query(
    ...     table_name='player_stats',
    ...     column_name='points',
    ...     window_sizes=[3, 5],
    ...     agg_function='AVG'
    ... )
    >>> con.execute(query).fetchdf()
    """
    # Validate inputs
    agg_function = agg_function.upper()
    if not window_sizes:
        raise ValueError("window_sizes cannot be empty")

    # Define window frame
    if include_current:
        window_frame = "ROWS BETWEEN {n} PRECEDING AND CURRENT ROW"
    else:
        window_frame = "ROWS BETWEEN {n} PRECEDING AND 1 PRECEDING"

    # Build base query
    query_parts = [
        f"SELECT",
        "    player_ID,",
        "    Gamecode,",
        f"    {column_name}",
    ]

    # Add window calculations for each size
    for window_size in window_sizes:
        # Overall statistic
        query_parts.append(f"""    
    , {agg_function}({column_name}) OVER (
        PARTITION BY player_id 
        ORDER BY Gamecode 
        {window_frame.format(n=window_size)}
    ) as {agg_function.lower()}_{column_name}_last_{window_size}_overall""")

        # Home games statistic
        query_parts.append(f"""    
    , {agg_function}(CASE WHEN home = true THEN {column_name} END) OVER (
        PARTITION BY player_id, home
        ORDER BY Gamecode 
        {window_frame.format(n=window_size)}
    ) as {agg_function.lower()}_{column_name}_last_{window_size}_home""")

        # Away games statistic
        query_parts.append(f"""    
    , {agg_function}(CASE WHEN home = false THEN {column_name} END) OVER (
        PARTITION BY player_id, home
        ORDER BY Gamecode 
        {window_frame.format(n=window_size)}
    ) as {agg_function.lower()}_{column_name}_last_{window_size}_away""")

    # Complete the query
    query_parts.append(f"""
FROM {table_name}
ORDER BY player_id, Gamecode""")

    return "\n".join(query_parts)


dset.columns

features = [
    "total_seconds",
    "Points",
    "FieldGoalsMade2",
    "FieldGoalsAttempted2",
    "FieldGoalsMade3",
    "FieldGoalsAttempted3",
    "FreeThrowsMade",
    "FreeThrowsAttempted",
    "OffensiveRebounds",
    "DefensiveRebounds",
    "TotalRebounds",
    "Assistances",
    "Steals",
    "Turnovers",
    "BlocksFavour",
    "BlocksAgainst",
    "FoulsCommited",
    "FoulsReceived",
    "Valuation",
    "Plusminus",
]

for feature in features:
    # Generate query for points averages
    query = generate_moving_stats_query(
        table_name="dset",
        column_name=feature,
        window_sizes=[3, 5, 10],
        agg_function="AVG",
        include_current=False,
    )

    # Execute query and get results as pandas DataFrame
    df_with_features = duckdb.sql(query).fetchdf()

    # Write the table to a Parquet file
    duckdb.sql(
        f"COPY df_with_features TO  (FORMAT 'euroleague_data/procesed/{feature}.parquet');"
    )
