from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rasta.web.accounts import Accounts, SESSION_COOKIE
from rasta.web.app import create_app

HEADERS = {"X-Rasta-Request": "1"}
INITIAL = "initial-test-password"
UPDATED = "updated-test-password"


def login(client, username, password=INITIAL):
    return client.post("/api/session/login", headers=HEADERS,
                       json={"username": username, "password": password})


def test_super_admin_manages_admins_and_revokes_sessions(tmp_path):
    store = Accounts(tmp_path / "accounts.sqlite3")
    root = store.create("superadmin", INITIAL, "super_admin")
    app = create_app(accounts=store)
    with TestClient(app) as supervisor, TestClient(app) as employee:
        assert employee.get("/api/products").status_code == 401
        assert employee.post("/api/session/login", json={"username": "superadmin", "password": INITIAL}).status_code == 403
        response = login(supervisor, "superadmin")
        assert response.status_code == 200 and response.json()["user"]["role"] == "super_admin"
        assert SESSION_COOKIE in response.cookies and "HttpOnly" in response.headers["set-cookie"]
        response = supervisor.post("/api/team", headers=HEADERS, json={"username": "operator", "password": INITIAL})
        assert response.status_code == 201
        admin_id = response.json()["id"]
        assert login(employee, "operator").status_code == 200
        assert employee.get("/api/team").status_code == 403
        assert employee.post("/api/team", headers=HEADERS, json={"username": "blocked", "password": INITIAL}).status_code == 403
        assert supervisor.patch(f"/api/team/{root['id']}", headers=HEADERS, json={"active": False}).status_code == 404
        assert supervisor.patch(f"/api/team/{admin_id}", headers=HEADERS, json={"active": False}).status_code == 200
        assert employee.get("/api/session").json()["user"] is None
        assert login(employee, "operator").status_code == 401
        assert supervisor.patch(f"/api/team/{admin_id}", headers=HEADERS,
                                json={"active": True, "password": UPDATED}).status_code == 200
        assert login(employee, "operator", UPDATED).status_code == 200
        assert employee.post("/api/session/password", headers=HEADERS,
                             json={"current": UPDATED, "password": INITIAL}).status_code == 200
        assert employee.get("/api/session").json()["user"] is None
        assert login(employee, "operator").status_code == 200
        token = employee.cookies.get(SESSION_COOKIE)
        assert employee.post("/api/session/logout", headers=HEADERS).status_code == 200
        assert store.session(token) is None
    assert len(Accounts(tmp_path / "accounts.sqlite3").list()) == 2
