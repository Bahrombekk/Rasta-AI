"""Provision a Rasta account using a password entered at the terminal."""
import argparse
from getpass import getpass
import os

import _bootstrap  # noqa: F401

from rasta.config import resolve_path
from rasta_backend.accounts import Accounts, ROLES
from rasta_backend.database import Database


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("username")
    parser.add_argument("--role", choices=ROLES, default="super_admin")
    parser.add_argument("--db", help="Explicit SQLite path, takes precedence over RASTA_DATABASE_URL")
    args = parser.parse_args()
    password = getpass("Yangi parol: ")
    if not password or getpass("Parolni takrorlang: ") != password:
        parser.error("Parollar mos kelmadi")
    url = os.environ.get("RASTA_DATABASE_URL")
    target = url if url and args.db is None else resolve_path(
        args.db or os.environ.get("RASTA_ACCOUNTS_DB", "data/accounts/users.sqlite3"))
    database = Database(target)
    try:
        user = Accounts(database).create(args.username, password, args.role)
    except ValueError as exc:
        parser.error(str(exc))
    finally:
        database.close()
    print(f'{user["username"]} ({user["role"]}) yaratildi.')


if __name__ == "__main__":
    main()
