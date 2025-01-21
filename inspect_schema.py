import sys

import polars as pl

if len(sys.argv) != 2:
    print("Usage: python inspect_schema.py <parquet_file_path>")
    sys.exit(1)

file_path = sys.argv[1]

try:
    df = pl.read_parquet(file_path)
    print(f"Columns in {file_path}:")
    print(df.columns)
except Exception as e:
    print(f"Error reading {file_path}: {str(e)}")
    sys.exit(1)
