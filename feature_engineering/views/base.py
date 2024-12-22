"""Base SQL view definitions."""

from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SQLView:
    """Base class for SQL views"""
    
    def __init__(self, name: str, sql_file: str):
        self.name = name
        self.sql_file = sql_file
        
    def create(self, db, **format_args):
        """Create the view in the database"""
        sql_path = Path("sql_queries") / self.sql_file
        db.execute_sql_file(sql_path, **format_args)
        logger.info(f"Created {self.name} view")
