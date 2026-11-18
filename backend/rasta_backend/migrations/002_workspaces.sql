CREATE TABLE IF NOT EXISTS organizations (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1, created INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS members (
    user_id INTEGER PRIMARY KEY REFERENCES users(id),
    org_id INTEGER NOT NULL REFERENCES organizations(id),
    role TEXT NOT NULL CHECK(role IN ('org_admin','operator'))
);
CREATE INDEX IF NOT EXISTS members_org ON members(org_id);
CREATE TABLE IF NOT EXISTS stores (
    id INTEGER PRIMARY KEY, org_id INTEGER NOT NULL REFERENCES organizations(id),
    name TEXT NOT NULL, address TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY, org_id INTEGER NOT NULL REFERENCES organizations(id),
    author_id INTEGER NOT NULL REFERENCES users(id),
    store_id INTEGER REFERENCES stores(id), note TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK(status IN ('draft','submitted','approved','returned')),
    result TEXT NOT NULL, photo BLOB NOT NULL, created INTEGER NOT NULL,
    submitted INTEGER, reviewed INTEGER, reviewer_id INTEGER REFERENCES users(id),
    feedback TEXT NOT NULL DEFAULT '',
    products INTEGER NOT NULL, recognized INTEGER NOT NULL, gaps INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS reports_org_time ON reports(org_id, submitted DESC);
CREATE INDEX IF NOT EXISTS reports_platform_time ON reports(submitted DESC)
    WHERE submitted IS NOT NULL;
CREATE INDEX IF NOT EXISTS reports_author_time ON reports(org_id, author_id, created DESC);
CREATE INDEX IF NOT EXISTS reports_org_display_time
    ON reports(org_id, COALESCE(submitted,created) DESC);
CREATE TABLE IF NOT EXISTS report_events (
    id INTEGER PRIMARY KEY, report_id TEXT NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id), status TEXT NOT NULL,
    note TEXT NOT NULL, created INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS report_events_report ON report_events(report_id,id);
CREATE TABLE IF NOT EXISTS platform_events (
    id INTEGER PRIMARY KEY, org_id INTEGER NOT NULL REFERENCES organizations(id),
    actor_id INTEGER NOT NULL REFERENCES users(id), action TEXT NOT NULL,
    subject TEXT NOT NULL, created INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS platform_events_org ON platform_events(org_id,id);
CREATE UNIQUE INDEX IF NOT EXISTS organization_name_unique ON organizations(LOWER(name));
CREATE UNIQUE INDEX IF NOT EXISTS store_name_unique ON stores(org_id, LOWER(name));
