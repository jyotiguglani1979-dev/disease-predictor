"""
src/db.py
---------
Tiny SQLite helper. All cleaned data lives in one file: data/processed/heartai.db
"""
import sqlite3

import pandas as pd

from config import DB_PATH


def connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def save(df: pd.DataFrame, table: str) -> None:
    """Write a DataFrame to a table (replaces the table if it exists)."""
    with connect() as con:
        df.to_sql(table, con, if_exists="replace", index=False)
    print(f"  saved {len(df):,} rows x {df.shape[1]} cols -> table '{table}'")


def load(table: str) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql(f"SELECT * FROM {table}", con)


def tables() -> list[str]:
    with connect() as con:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    return [r[0] for r in rows]
