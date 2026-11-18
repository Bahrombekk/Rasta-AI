CREATE INDEX IF NOT EXISTS sessions_expiration ON sessions(expires);
CREATE INDEX IF NOT EXISTS login_attempts_expiration ON login_attempts(expires);
CREATE INDEX IF NOT EXISTS reports_org_store_submission ON reports(org_id, store_id, submitted DESC);
CREATE INDEX IF NOT EXISTS reports_org_status_submission ON reports(org_id, status, submitted DESC);
CREATE INDEX IF NOT EXISTS field_visits_org_start ON field_visits(org_id, started DESC);
CREATE INDEX IF NOT EXISTS field_visits_org_agent_start ON field_visits(org_id, agent_id, started DESC);
CREATE INDEX IF NOT EXISTS field_deliveries_visit_time ON field_deliveries(visit_id, created DESC);
CREATE INDEX IF NOT EXISTS field_tasks_org_completed ON field_tasks(org_id, completed DESC);
