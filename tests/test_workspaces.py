from fastapi.testclient import TestClient

from test_accounts import HEADERS, INITIAL, login
from test_web import FakePipe, jpg, web
from rasta.config import Config
from rasta.web.accounts import Accounts
from rasta.web.workspaces import Workspaces


def test_organization_boundaries_and_shelf_review(tmp_path, monkeypatch):
    accounts = Accounts(tmp_path / "accounts.sqlite3")
    accounts.create("root-admin", INITIAL, "super_admin")
    accounts.create("legacy-admin", INITIAL)
    spaces = Workspaces(accounts)
    fake = FakePipe()
    monkeypatch.setattr(web.State, "pipe", fake)
    monkeypatch.setattr(web.State, "images_dir", tmp_path / "images")
    app = web.create_app(Config({"catalog": {"path": ":memory:"}}), accounts=accounts, workspaces=spaces)
    image = {"image": ("s.jpg", jpg(), "image/jpeg")}
    packshot = [("files", ("s.jpg", jpg(), "image/jpeg"))]
    with (TestClient(app) as root, TestClient(app) as admin, TestClient(app) as other,
          TestClient(app) as operator):
        assert login(root, "root-admin").status_code == 200
        migrated = login(admin, "legacy-admin").json()["user"]
        assert (migrated["role"], migrated["org_id"]) == ("org_admin", 1)
        response = root.post("/api/organizations", headers=HEADERS, json={
            "name": "Another Company", "username": "foreign-admin", "password": INITIAL,
        })
        assert response.status_code == 201
        org_id = response.json()["id"]
        assert login(other, "foreign-admin").json()["user"]["org_id"] == org_id
        assert admin.get("/api/organizations").status_code == 403
        assert admin.get("/api/team").status_code == 404
        assert root.post("/api/analyze", headers=HEADERS, files=image).status_code == 403

        for client, brand in [(admin, "A"), (other, "B")]:
            assert client.post("/api/products", headers=HEADERS, data={
                "name": "Cola", "brand": brand,
            }, files=packshot).status_code == 200
        assert admin.get("/api/products/cola").json()["brand"] == "A"
        assert other.get("/api/products/cola").json()["brand"] == "B"
        assert (tmp_path / "images" / "cola" / "000.jpg").exists()
        assert (tmp_path / "images" / ".organizations" / str(org_id) / "cola" / "000.jpg").exists()
        assert admin.get("/media/cola/000.jpg").status_code == 200

        response = admin.post("/api/members", headers=HEADERS, json={"username": "field-user", "password": INITIAL})
        assert response.status_code == 201
        member_id = response.json()["id"]
        assert login(operator, "field-user").json()["user"]["role"] == "operator"
        assert operator.get("/api/products").status_code == 403
        assert operator.post("/api/products", headers=HEADERS, data={"name": "Test"}, files=packshot).status_code == 403
        assert other.patch(f"/api/members/{member_id}", headers=HEADERS, json={"active": False}).status_code == 404
        assert operator.post("/api/stores", headers=HEADERS, json={"name": "Blocked"}).status_code == 403

        store = admin.post("/api/stores", headers=HEADERS, json={"name": "Tashkent"}).json()["id"]
        foreign_store = other.post("/api/stores", headers=HEADERS, json={"name": "Foreign"}).json()["id"]
        assert other.patch(f"/api/stores/{store}", headers=HEADERS, json={"name": "Blocked"}).status_code == 404
        assert admin.patch(f"/api/stores/{store}", headers=HEADERS, json={"name": "Tashkent Shop"}).status_code == 200
        assert [row["id"] for row in operator.get("/api/stores").json()["stores"]] == [store]
        response = operator.post("/api/analyze", headers=HEADERS, files=image)
        assert response.status_code == 200
        report = "/api/reports/" + response.json()["report_id"]
        assert admin.get(report).status_code == 404
        assert operator.post(report + "/submit", headers=HEADERS, json={"store_id": foreign_store}).status_code == 400
        assert operator.post(report + "/submit", headers=HEADERS, json={"store_id": store}).status_code == 200
        assert operator.post(report + "/submit", headers=HEADERS, json={"store_id": store}).status_code == 200
        assert other.get(report).status_code == 404
        assert other.get(report + "/photo").status_code == 404
        assert admin.get(report + "/photo").headers["content-type"] == "image/jpeg"
        assert operator.post(report + "/review", headers=HEADERS, json={"status": "approved"}).status_code == 403
        assert admin.post(report + "/review", headers=HEADERS, json={
            "status": "returned", "feedback": "Try again",
        }).status_code == 200
        assert operator.get(report).json()["feedback"] == "Try again"
        assert operator.post(report + "/submit", headers=HEADERS, json={"store_id": store}).status_code == 200
        assert admin.post(report + "/review", headers=HEADERS, json={"status": "approved"}).status_code == 200
        assert [event["status"] for event in admin.get(report).json()["events"]] == [
            "approved", "submitted", "returned", "submitted",
        ]
        assert admin.get("/api/overview?days=7").json()["approved"] == 1
        assert other.get("/api/overview?days=7").json()["reports"] == 0
        assert root.patch(f"/api/organizations/{org_id}", headers=HEADERS,
                          json={"name": "Another Company", "active": False}).status_code == 200
        assert other.get("/api/products").status_code == 401
