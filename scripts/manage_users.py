"""Provision a Rasta account using a password entered at the terminal."""
import argparse
from getpass import getpass
import os

import _bootstrap  # noqa: F401

from rasta.config import resolve_path
from rasta.web.accounts import Accounts, ROLES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("username")
    parser.add_argument("--role", choices=ROLES, default="super_admin")
    parser.add_argument("--db", default=os.environ.get("RASTA_ACCOUNTS_DB", "data/accounts/users.sqlite3"))
    args = parser.parse_args()
    password = getpass("Yangi parol: ")
    if not password or getpass("Parolni takrorlang: ") != password:
        parser.error("Parollar mos kelmadi")
    try:
        user = Accounts(resolve_path(args.db)).create(args.username, password, args.role)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'{user["username"]} ({user["role"]}) yaratildi.')


if __name__ == "__main__":
    main()
