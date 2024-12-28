"""Database connection and setup module."""

import logging
from pathlib import Path
import duckdb
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureDatabase:
    """Handles database connections and SQL execution"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self):
        """Initialize DuckDB connection with custom functions"""
        logger.info("Initializing DuckDB connection")
        self.conn = duckdb.connect(self.db_path)

        # Register custom functions
        self.conn.execute("""
            CREATE MACRO convert_minutes(time_str) AS (
                CASE 
                    WHEN time_str = '' OR time_str IS NULL OR time_str = 'DNP' THEN 0
                    ELSE CAST(SPLIT_PART(time_str, ':', 1) AS INTEGER) + 
                         CAST(SPLIT_PART(time_str, ':', 2) AS FLOAT) / 60
                END
            );
        """)

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def execute_sql_file(self, sql_path: Path, **format_args) -> None:
        """Execute SQL from file with formatting"""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")

        if not sql_path.exists():
            raise FileNotFoundError(f"SQL file not found: {sql_path}")

        sql = sql_path.read_text()
        if format_args:
            sql = sql.format(**format_args)

        self.conn.execute(sql)
        logger.info(f"Executed SQL from {sql_path}")

    def query_to_df(self, query: str):
        """Execute query and return pandas DataFrame"""
        if not self.conn:
            raise RuntimeError("Database connection not initialized")
        return self.conn.execute(query).df()
