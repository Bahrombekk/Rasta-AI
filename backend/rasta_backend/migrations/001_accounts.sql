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
