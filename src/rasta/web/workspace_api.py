import csv
from dataclasses import dataclass
from datetime import date, datetime
import io
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from .session_api import AdminUpdate, Login
from .workspaces import TASHKENT

ReportStatus = Literal["draft", "submitted", "approved", "returned"]
Period = Literal["7", "30", "90"]
CSV_LIMIT = 10_000


@dataclass(frozen=True)
class ReportFilters:
    status: ReportStatus | None
    store_id: int | None
    author_id: int | None
    start: date | None
    end: date | None


def report_filters(status: ReportStatus | None = None, store_id: int | None = Query(None, gt=0),
                   author_id: int | None = Query(None, gt=0),
                   start: date | None = Query(None, alias="from"), end: date | None = Query(None, alias="to")):
    if start and end and start > end:
        raise HTTPException(400, "Boshlanish sanasi tugash sanasidan keyin bo‘lishi mumkin emas.")
    return ReportFilters(status, store_id, author_id, start, end)


def csv_text(value):
    value = value or ""
    if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


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

    @api.get("/platform/overview")
    def platform_overview(request: Request, days: Period = "30"):
        require(request, "super_admin")
        return workspaces.platform_overview(int(days))

    @api.get("/organizations/{org_id}/overview")
    def organization_overview(request: Request, org_id: int, days: Period = "30"):
        require(request, "super_admin")
        overview = workspaces.platform_overview(int(days), org_id)
        if overview is None:
            raise HTTPException(404, "Korxona topilmadi.")
        return overview

    @api.get("/organizations")
    def organizations(request: Request):
        require(request, "super_admin")
        return {"organizations": workspaces.organizations()}

    @api.post("/organizations", status_code=201)
    def create_org(request: Request, body: Organization):
        actor = require(request, "super_admin")
        return validate(workspaces.create_org, body.name, body.username, body.password, actor["id"])

    @api.patch("/organizations/{org_id}")
    def update_org(request: Request, org_id: int, body: OrganizationUpdate):
        actor = require(request, "super_admin")
        if not validate(workspaces.update_org, org_id, body.name, body.active, actor["id"]):
            raise HTTPException(404, "Korxona topilmadi.")
        return {"ok": True}

    @api.get("/organizations/{org_id}/members")
    def org_admins(request: Request, org_id: int):
        require(request, "super_admin")
        return {"users": [u for u in workspaces.members(org_id) if u["role"] == "org_admin"]}

    @api.post("/organizations/{org_id}/members", status_code=201)
    def add_org_admin(request: Request, org_id: int, body: Login):
        actor = require(request, "super_admin")
        return validate(workspaces.create_member, org_id, body.username, body.password, "org_admin", actor["id"])

    @api.patch("/organizations/{org_id}/members/{user_id}")
    def edit_org_admin(request: Request, org_id: int, user_id: int, body: AdminUpdate):
        actor = require(request, "super_admin")
        if not validate(workspaces.update_member, org_id, user_id, "org_admin", body.active, body.password, actor["id"]):
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
    def overview(request: Request, days: Period = "30"):
        return workspaces.overview(require(request, "org_admin", "operator"), int(days))

    @api.get("/reports")
    def reports(request: Request, filters: ReportFilters = Depends(report_filters),
                offset: int = Query(0, ge=0, le=2_147_483_647)):
        user = require(request, "org_admin", "operator")
        rows, total = workspaces.list_reports(user, filters, offset)
        return {"reports": rows, "total": total, "has_more": offset + len(rows) < total}

    @api.get("/reports/export.csv")
    def export_reports(request: Request, filters: ReportFilters = Depends(report_filters)):
        user = require(request, "org_admin")
        if filters.status == "draft":
            raise HTTPException(400, "CSV formatida yuborilgan auditlar yuklanadi.")
        rows, _ = workspaces.list_reports(user, filters, limit=CSV_LIMIT + 1, count_total=False)
        if len(rows) > CSV_LIMIT:
            raise HTTPException(400, "CSV uchun davrni qisqartiring (ko‘pi bilan 10 000 audit).")
        output = io.StringIO(newline="")
        output.write("\ufeff")
        writer = csv.writer(output)
        writer.writerow(["Audit ID", "Do‘kon", "Yuborgan", "Holat", "Yuborilgan vaqt (Toshkent)",
                         "Topilgan mahsulotlar", "Tanilgan SKU", "Bo‘sh joylar"])
        statuses = {"submitted": "Tekshiruvda", "approved": "Tasdiqlangan", "returned": "Qaytarilgan"}
        for row in rows:
            timestamp = datetime.fromtimestamp(row["submitted"], TASHKENT).strftime("%d.%m.%Y %H:%M")
            writer.writerow([row["id"], csv_text(row["store"]), csv_text(row["author"]), statuses[row["status"]],
                             timestamp, row["products"], row["recognized"], row["gaps"]])
        return Response(output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="rasta-auditlar.csv"'})

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
