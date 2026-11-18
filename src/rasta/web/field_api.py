from datetime import date
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .workspace_api import Period, require, validate


class Assignments(BaseModel):
    store_ids: list[int] = Field(max_length=1000)


class Start(BaseModel):
    store_id: int = Field(gt=0)


class Note(BaseModel):
    note: str = Field(default="", max_length=1000)


class Delivery(BaseModel):
    request_key: UUID
    sku_id: str = Field(min_length=1, max_length=120)
    quantity: int = Field(gt=0, le=1_000_000)


class Task(BaseModel):
    store_id: int = Field(gt=0)
    agent_id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=160)
    kind: Literal["audit", "delivery", "other"]
    due: date | None = None


class CompleteTask(Note):
    visit_id: int = Field(gt=0)


def router(field, get_pipe, catalog_lock):
    api = APIRouter(prefix="/api/field")

    @api.get("")
    def workspace(request: Request):
        return field.snapshot(require(request, "org_admin", "operator"))

    @api.get("/overview")
    def overview(request: Request, days: Period = "30", store_id: int | None = None):
        user = require(request, "org_admin")
        data = field.overview(user["org_id"], int(days), store_id)
        if store_id is not None and not data["stores"]:
            raise HTTPException(404, "Do‘kon topilmadi.")
        return data

    @api.get("/agents/{agent_id}/stores")
    def assignments(request: Request, agent_id: int):
        user = require(request, "org_admin")
        store_ids = field.assignments(user["org_id"], agent_id)
        if store_ids is None:
            raise HTTPException(404, "Agent topilmadi.")
        return {"store_ids": store_ids}

    @api.put("/agents/{agent_id}/stores")
    def edit_assignments(request: Request, agent_id: int, body: Assignments):
        user = require(request, "org_admin")
        store_ids = validate(field.replace_assignments, user["org_id"], agent_id, body.store_ids)
        if store_ids is None:
            raise HTTPException(404, "Agent topilmadi.")
        return {"store_ids": store_ids}

    @api.post("/visits", status_code=201)
    def start(request: Request, body: Start):
        return {"id": validate(field.start_visit, require(request, "operator"), body.store_id)}

    @api.get("/visits/{visit_id}")
    def visit(request: Request, visit_id: int):
        detail = field.visit(require(request, "org_admin", "operator"), visit_id)
        if detail is None:
            raise HTTPException(404, "Tashrif topilmadi.")
        return detail

    @api.post("/visits/{visit_id}/finish")
    def finish(request: Request, visit_id: int, body: Note):
        if not validate(field.finish_visit, require(request, "operator"), visit_id, body.note):
            raise HTTPException(404, "Ochiq tashrif topilmadi.")
        return {"ok": True}

    @api.get("/visits/{visit_id}/products")
    def products(request: Request, visit_id: int):
        if field.own_visit(require(request, "operator"), visit_id) is None:
            raise HTTPException(404, "Ochiq tashrif topilmadi.")
        pipe = get_pipe()
        with catalog_lock:
            return {"products": [{"sku_id": sku, "name": info["name"], "brand": info.get("brand"),
                                  "barcode": info.get("barcode")}
                                 for sku, info in pipe.catalog.sku_info().items()]}

    @api.post("/visits/{visit_id}/deliveries", status_code=201)
    def delivery(request: Request, visit_id: int, body: Delivery):
        user = require(request, "operator")
        # Check ownership before opening the catalog. A retry can return a saved line after closing the visit.
        visit = field.visit(user, visit_id)
        if visit is None:
            raise HTTPException(404, "Tashrif topilmadi.")
        pipe = get_pipe()
        with catalog_lock:
            info = pipe.catalog.sku_info().get(body.sku_id)
        delivery_id = validate(field.record_delivery, user, visit_id, str(body.request_key), body.sku_id,
                               info["name"] if info else None, body.quantity)
        if delivery_id is None:
            raise HTTPException(404, "Ochiq tashrif topilmadi.")
        return {"id": delivery_id}

    @api.post("/tasks", status_code=201)
    def create_task(request: Request, body: Task):
        user = require(request, "org_admin")
        return {"id": validate(field.create_task, user["org_id"], body.agent_id, body.store_id, body.title,
                               body.kind, body.due.isoformat() if body.due else None)}

    @api.post("/tasks/{task_id}/complete")
    def finish_task(request: Request, task_id: int, body: CompleteTask):
        if not validate(field.finish_task, require(request, "operator"), task_id, body.visit_id, body.note):
            raise HTTPException(404, "Tashrifdagi ochiq topshiriq topilmadi.")
        return {"ok": True}

    @api.post("/tasks/{task_id}/cancel")
    def cancel(request: Request, task_id: int):
        user = require(request, "org_admin")
        if not field.cancel_task(user["org_id"], task_id):
            raise HTTPException(404, "Ochiq topshiriq topilmadi.")
        return {"ok": True}

    return api
