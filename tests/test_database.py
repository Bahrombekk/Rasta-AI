import sqlite3

import pytest

from rasta_backend.accounts import Accounts
from rasta_backend.database import Database, INTEGRITY_ERRORS
from rasta_backend.migrate_sqlite import migrate
from rasta_backend.workspaces import Workspaces


def test_existing_sqlite_file_migrates_and_preserves_workspace(tmp_path, postgres_url):
    source = tmp_path / "workspace.sqlite3"
    old = Accounts(source)
    old_admin = old.create("first-admin", "long-testing-password")
    workspace = Workspaces(old)
    organization = workspace.create_org("Other company", "other-admin", "long-testing-password")
    org_id = organization["id"]
    store = workspace.create_store(1, "Shop one", "")
    elsewhere = workspace.create_store(org_id, "Shop two", "")
    agent = workspace.create_member(1, "field-agent", "long-testing-password", "operator")
    workspace.field.replace_assignments(1, agent["id"], [store["id"]])
    actor = workspace.identity(agent)
    visit = workspace.field.start_visit(actor, store["id"])
    report = workspace.create_draft(actor, {"summary": {"num_products": 2, "num_known": 1, "oos_gaps": 0}},
                                    b"photo-binary", visit_id=visit)
    old_token, _ = old.login("first-admin", "long-testing-password", "local")

    target = Database(postgres_url)
    try:
        counts = migrate(source, target)
        assert counts["users"] == 3 and counts["field_visits"] == 1 and counts["reports"] == 1
        migrated = Accounts(target)
        stored = Workspaces(migrated)
        assert migrated.session(old_token) == old_admin
        assert stored.report(actor, report, photo=True) == b"photo-binary"
        assert stored.field.visit(actor, visit)["store"] == "Shop one"
        assert stored.create_store(1, "New shop", "")["id"] > elsewhere["id"]
        with pytest.raises(INTEGRITY_ERRORS):
            with target.connect() as conn:
                conn.execute("INSERT INTO store_assignments VALUES(?,?,?)", (1, agent["id"], elsewhere["id"]))
        with pytest.raises(ValueError, match="already contains"):
            migrate(source, target)
    finally:
        target.close()
    with sqlite3.connect(source) as database:
        assert database.execute("SELECT COUNT(*) FROM stores").fetchone() == (2,)
