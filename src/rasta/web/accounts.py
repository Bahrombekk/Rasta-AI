"""Persistent accounts and revocable browser sessions."""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import pbkdf2_hmac, sha256
import hmac
from pathlib import Path
import re
import secrets
import sqlite3
import time

SESSION_COOKIE = "rasta_session"
SESSION_TTL = 12 * 60 * 60
LOGIN_WINDOW = 15 * 60
LOGIN_LIMIT = 8
CLIENT_LOGIN_LIMIT = 60
PASSWORD_ROUNDS = 600_000
ROLES = ("admin", "super_admin")


def username_key(username):
    username = username.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,47}", username):
        raise ValueError("Login 3–48 ta lotin harfi, raqam, nuqta, pastki chiziq yoki tiredan iborat bo‘lsin.")
    return username


def new_password(password):
    if not 12 <= len(password) <= 128:
        raise ValueError("Parol 12–128 ta belgidan iborat bo‘lsin.")
    return password_hash(password, secrets.token_bytes(16))


def password_hash(password, salt):
    return salt + pbkdf2_hmac("sha256", password.encode(), salt, PASSWORD_ROUNDS)


def token_key(token):
    return sha256(token.encode()).digest()


class Accounts:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript("""
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE, password BLOB NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin','super_admin')),
                    active INTEGER NOT NULL DEFAULT 1, created INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token BLOB PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
                    expires INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
                CREATE TABLE IF NOT EXISTS login_attempts (
                    key TEXT PRIMARY KEY, expires INTEGER NOT NULL, attempts INTEGER NOT NULL
                );
            """)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _user(row):
        return {"id": row["id"], "username": row["username"], "role": row["role"],
                "active": bool(row["active"])}

    def list(self):
        with self._db() as db:
            return [self._user(row) for row in db.execute("SELECT * FROM users ORDER BY role DESC, username")]

    def create(self, username, password, role="admin"):
        username = username_key(username)
        if role not in ROLES:
            raise ValueError("Noto‘g‘ri ruxsat turi.")
        hashed = new_password(password)
        with self._db() as db:
            try:
                cursor = db.execute(
                    "INSERT INTO users(username,password,role,created) VALUES (?,?,?,?)",
                    (username, hashed, role, int(time.time())),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Bu login band.") from exc
            return self._user(db.execute("SELECT * FROM users WHERE id=?", (cursor.lastrowid,)).fetchone())

    def login(self, username, password, client):
        try:
            username = username_key(username)
        except ValueError:
            return None, False
        now = int(time.time())
        keys = [(f"ip/{client}", CLIENT_LOGIN_LIMIT), (f"user/{client}/{username}", LOGIN_LIMIT)]
        with self._db() as db:
            db.execute("DELETE FROM login_attempts WHERE expires<=?", (now,))
            for key, limit in keys:
                row = db.execute("SELECT attempts FROM login_attempts WHERE key=?", (key,)).fetchone()
                if row and row["attempts"] >= limit:
                    return None, True
            db.executemany("""
                INSERT INTO login_attempts VALUES (?,?,1)
                ON CONFLICT(key) DO UPDATE SET attempts=attempts+1
                """, [(key, now + LOGIN_WINDOW) for key, _ in keys])
            user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        stored = user["password"] if user else bytes(48)
        matched = hmac.compare_digest(stored, password_hash(password, stored[:16]))
        if not matched or not user or not user["active"]:
            return None, False
        token = secrets.token_urlsafe(32)
        with self._db() as db:
            db.execute("DELETE FROM sessions WHERE expires<=?", (now,))
            current = db.execute("SELECT * FROM users WHERE id=?", (user["id"],)).fetchone()
            if not current["active"] or current["password"] != stored:
                return None, False
            db.execute("DELETE FROM login_attempts WHERE key=?", (keys[1][0],))
            db.execute("INSERT INTO sessions VALUES (?,?,?)", (token_key(token), user["id"], now + SESSION_TTL))
        return token, False

    def session(self, token):
        if not token or len(token) > 100:
            return None
        with self._db() as db:
            row = db.execute("""
                SELECT users.* FROM sessions JOIN users ON users.id=sessions.user_id
                WHERE sessions.token=? AND sessions.expires>? AND users.active=1
                """, (token_key(token), int(time.time()))).fetchone()
        return self._user(row) if row else None

    def logout(self, token):
        if token and len(token) <= 100:
            with self._db() as db:
                db.execute("DELETE FROM sessions WHERE token=?", (token_key(token),))

    def edit_admin(self, user_id, active, password=None):
        hashed = new_password(password) if password is not None else None
        with self._db() as db:
            cursor = db.execute("""
                UPDATE users SET active=?, password=COALESCE(?, password)
                WHERE id=? AND role='admin'
                """, (bool(active), hashed, user_id))
            if cursor.rowcount != 1:
                return None
            if not active or hashed is not None:
                db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            return self._user(db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())

    def change_password(self, user_id, old, new):
        hashed = new_password(new)
        with self._db() as db:
            row = db.execute("SELECT password FROM users WHERE id=? AND active=1", (user_id,)).fetchone()
            if not row or not hmac.compare_digest(row["password"], password_hash(old, row["password"][:16])):
                return False
            cursor = db.execute("UPDATE users SET password=? WHERE id=? AND password=? AND active=1",
                                (hashed, user_id, row["password"]))
            if cursor.rowcount != 1:
                return False
            db.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
            return True
