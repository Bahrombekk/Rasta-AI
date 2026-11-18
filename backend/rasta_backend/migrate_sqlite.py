"""Import an existing SQLite workspace into an empty PostgreSQL database."""
import argparse
import os
from pathlib import Path
import sqlite3
from contextlib import closing

from psycopg import sql

from .accounts import Accounts
from .database import Database
from .workspaces import Workspaces

TABLES = (
    "users", "organizations", "members", "stores", "field_visits", "reports",
    "report_events", "platform_events", "store_assignments", "field_deliveries",
    "field_tasks", "sessions", "login_attempts",
)
IDENTITIES = ("users", "organizations", "stores", "field_visits", "report_events",
              "platform_events", "field_deliveries", "field_tasks")


def migrate(source_file, target):
    if not target.postgres:
        raise ValueError("The destination must be PostgreSQL.")
    source = Path(source_file).resolve(strict=True).as_uri() + "?mode=ro"
    with closing(sqlite3.connect(source, uri=True)) as old:
        old.execute("BEGIN")
        if old.execute("PRAGMA quick_check").fetchone()[0] != "ok" or old.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("The SQLite source did not pass the integrity checks.")
        available = {row[0] for row in old.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "users" not in available:
            raise ValueError("The source is missing the Rasta users table.")
        Workspaces(Accounts(target), seed=False)
        imported = {}
        with target.connect() as db:
            db.write_lock()
            for table in TABLES:
                identifier = sql.Identifier(table)
                if db.raw.execute(sql.SQL("SELECT 1 FROM {} LIMIT 1").format(identifier)).fetchone():
                    raise ValueError("The PostgreSQL destination already contains workspace data.")
                if table not in available:
                    continue
                rows = old.execute(f"SELECT * FROM {table}")
                fields = sql.SQL(", ").join(sql.Identifier(column[0]) for column in rows.description)
                copied = 0
                with db.raw.cursor().copy(sql.SQL("COPY {} ({}) FROM STDIN").format(identifier, fields)) as operation:
                    while batch := rows.fetchmany(8):
                        for row in batch:
                            operation.write_row(row)
                        copied += len(batch)
                stored = db.raw.execute(sql.SQL("SELECT COUNT(*) AS total FROM {}").format(identifier)).fetchone()["total"]
                if stored != copied:
                    raise ValueError(f"Unexpected row count for {table}.")
                imported[table] = copied
            for table in IDENTITIES:
                db.advance_identity(table)
        return imported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    args = parser.parse_args()
    url = os.getenv("RASTA_DATABASE_URL")
    if not url or not url.startswith(("postgresql://", "postgres://")):
        parser.error("Set RASTA_DATABASE_URL to an empty PostgreSQL database.")
    destination = Database(url)
    try:
        counts = migrate(args.sqlite, destination)
    finally:
        destination.close()
    for table, count in counts.items():
        print(f"{table}: {count}")


if __name__ == "__main__":
    main()
