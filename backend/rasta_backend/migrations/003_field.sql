CREATE TABLE IF NOT EXISTS store_assignments (
    org_id INTEGER NOT NULL REFERENCES organizations(id),
    agent_id INTEGER NOT NULL REFERENCES users(id),
    store_id INTEGER NOT NULL REFERENCES stores(id),
    PRIMARY KEY(org_id,agent_id,store_id)
);
CREATE INDEX IF NOT EXISTS assignments_stores ON store_assignments(org_id,store_id);
CREATE TABLE IF NOT EXISTS field_visits (
    id INTEGER PRIMARY KEY, org_id INTEGER NOT NULL REFERENCES organizations(id),
    store_id INTEGER NOT NULL REFERENCES stores(id), agent_id INTEGER NOT NULL REFERENCES users(id),
    started INTEGER NOT NULL, finished INTEGER, note TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS field_active_agent ON field_visits(org_id,agent_id) WHERE finished IS NULL;
CREATE INDEX IF NOT EXISTS field_store_time ON field_visits(org_id,store_id,started DESC);
CREATE TABLE IF NOT EXISTS field_deliveries (
    id INTEGER PRIMARY KEY, visit_id INTEGER NOT NULL REFERENCES field_visits(id),
    request_key TEXT NOT NULL, sku_id TEXT NOT NULL, product TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK(quantity>0), created INTEGER NOT NULL, UNIQUE(visit_id,request_key)
);
CREATE TABLE IF NOT EXISTS field_tasks (
    id INTEGER PRIMARY KEY, org_id INTEGER NOT NULL REFERENCES organizations(id),
    store_id INTEGER NOT NULL REFERENCES stores(id), agent_id INTEGER NOT NULL REFERENCES users(id),
    title TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('audit','delivery','other')),
    due TEXT, status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','done','cancelled')),
    visit_id INTEGER REFERENCES field_visits(id), note TEXT NOT NULL DEFAULT '', completed INTEGER
);
CREATE INDEX IF NOT EXISTS field_tasks_agent ON field_tasks(org_id,agent_id,status);
