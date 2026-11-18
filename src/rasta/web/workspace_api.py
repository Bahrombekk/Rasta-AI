from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from .session_api import AdminUpdate, Login


class Organization(Login):
    name: str = Field(max_length=100)


class OrganizationUpdate(BaseModel):
    name: str = Field(max_length=100)
    active: bool


class Store(BaseModel):
    name: str = Field(max_length=100)
    address: str = Field(default="", max_length=200)


class Submit(BaseModel):
    store_id: int
    note: str = Field(default="", max_length=1000)


class Review(BaseModel):
    status: Literal["approved", "returned"]
    feedback: str = Field(default="", max_length=1000)


def require(request, *roles):
    user = request.state.user
    if not user or user["role"] not in roles:
        raise HTTPException(403, "Ushbu amal uchun ruxsat yetarli emas.")
    return user


def validate(action, *args):
    try:
        return action(*args)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def router(workspaces):
    api = APIRouter(prefix="/api")

    @api.get("/organizations")
    def organizations(request: Request):
        require(request, "super_admin")
        return {"organizations": workspaces.organizations()}

    @api.post("/organizations", status_code=201)
    def create_org(request: Request, body: Organization):
        require(request, "super_admin")
        return validate(workspaces.create_org, body.name, body.username, body.password)

    @api.patch("/organizations/{org_id}")
    def update_org(request: Request, org_id: int, body: OrganizationUpdate):
        require(request, "super_admin")
        if not validate(workspaces.update_org, org_id, body.name, body.active):
            raise HTTPException(404, "Korxona topilmadi.")
        return {"ok": True}

    @api.get("/organizations/{org_id}/members")
    def org_admins(request: Request, org_id: int):
        require(request, "super_admin")
        return {"users": [u for u in workspaces.members(org_id) if u["role"] == "org_admin"]}

    @api.post("/organizations/{org_id}/members", status_code=201)
    def add_org_admin(request: Request, org_id: int, body: Login):
        require(request, "super_admin")
        return validate(workspaces.create_member, org_id, body.username, body.password, "org_admin")

    @api.patch("/organizations/{org_id}/members/{user_id}")
    def edit_org_admin(request: Request, org_id: int, user_id: int, body: AdminUpdate):
        require(request, "super_admin")
        if not validate(workspaces.update_member, org_id, user_id, "org_admin", body.active, body.password):
            raise HTTPException(404, "Korxona admini topilmadi.")
        return {"ok": True}

    @api.get("/members")
    def members(request: Request):
        user = require(request, "org_admin")
        return {"users": workspaces.members(user["org_id"])}

    @api.post("/members", status_code=201)
    def add_member(request: Request, body: Login):
        user = require(request, "org_admin")
        return validate(workspaces.create_member, user["org_id"], body.username, body.password, "operator")

    @api.patch("/members/{user_id}")
    def edit_member(request: Request, user_id: int, body: AdminUpdate):
        user = require(request, "org_admin")
        if not validate(workspaces.update_member, user["org_id"], user_id, "operator", body.active, body.password):
            raise HTTPException(404, "Xodim topilmadi.")
        return {"ok": True}

    @api.get("/stores")
    def stores(request: Request):
        user = require(request, "org_admin", "operator")
        return {"stores": workspaces.stores(user["org_id"])}

    @api.post("/stores", status_code=201)
    def add_store(request: Request, body: Store):
        user = require(request, "org_admin")
        return validate(workspaces.create_store, user["org_id"], body.name, body.address)

    @api.patch("/stores/{store_id}")
    def edit_store(request: Request, store_id: int, body: Store):
        user = require(request, "org_admin")
        if not validate(workspaces.update_store, user["org_id"], store_id, body.name, body.address):
            raise HTTPException(404, "Do‘kon topilmadi.")
        return {"ok": True}

    @api.get("/overview")
    def overview(request: Request, days: Literal["7", "30", "90"] = "30"):
        return workspaces.overview(require(request, "org_admin", "operator"), int(days))

    @api.get("/reports")
    def reports(request: Request, status: Literal["draft", "submitted", "approved", "returned"] | None = None,
                offset: int = Query(0, ge=0)):
        user = require(request, "org_admin", "operator")
        rows = workspaces.list_reports(user, status, offset)
        return {"reports": rows[:50], "has_more": len(rows) > 50}

    @api.get("/reports/{report_id}")
    def report(request: Request, report_id: str):
        data = workspaces.report(require(request, "org_admin", "operator"), report_id)
        if data is None:
            raise HTTPException(404, "Hisobot topilmadi.")
        return data

    @api.get("/reports/{report_id}/photo")
    def photo(request: Request, report_id: str):
        data = workspaces.report(require(request, "org_admin", "operator"), report_id, photo=True)
        if data is None:
            raise HTTPException(404, "Rasm topilmadi.")
        return Response(data, media_type="image/jpeg")

    @api.post("/reports/{report_id}/submit")
    def submit(request: Request, report_id: str, body: Submit):
        user = require(request, "org_admin", "operator")
        status = validate(workspaces.submit, user, report_id, body.store_id, body.note)
        if status is None:
            raise HTTPException(404, "Tahlil topilmadi.")
        return {"id": report_id, "status": status}

    @api.post("/reports/{report_id}/review")
    def review(request: Request, report_id: str, body: Review):
        user = require(request, "org_admin")
        if body.status == "returned" and not body.feedback.strip():
            raise HTTPException(400, "Qaytarish sababini kiriting.")
        if workspaces.report(user, report_id) is None:
            raise HTTPException(404, "Hisobot topilmadi.")
        if not workspaces.review(user, report_id, body.status, body.feedback):
            raise HTTPException(409, "Hisobot hozir ko‘rib chiqish navbatida emas.")
        return {"id": report_id, "status": body.status}

    return api
