"""Browser identity and super-admin user management routes."""
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from .accounts import SESSION_COOKIE, SESSION_TTL


class Login(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=128)


class AdminUpdate(BaseModel):
    active: bool
    password: str | None = Field(default=None, max_length=128)


class PasswordUpdate(BaseModel):
    current: str = Field(max_length=128)
    password: str = Field(max_length=128)


def session_router(accounts, secure, *, workspaces=None):
    router = APIRouter()

    def cookie_secure(request: Request) -> bool:
        # RASTA_SECURE_COOKIES bilan majburlanadi; bundan tashqari so'rov HTTPS
        # orqali kelsa (proksi X-Forwarded-Proto'si) avtomatik Secure bo'ladi.
        return secure or request.url.scheme == "https"

    @router.get("/api/session")
    def session(request: Request, response: Response):
        response.headers["Cache-Control"] = "no-store"
        return {"enabled": accounts is not None, "workspaces": workspaces is not None, "user": request.state.user}

    @router.post("/api/session/login")
    def login(body: Login, request: Request, response: Response):
        if accounts is None:
            raise HTTPException(404, "Hisoblar o‘chirilgan")
        client = request.client.host if request.client else "unknown"
        token, limited = accounts.login(body.username, body.password, client)
        if limited:
            raise HTTPException(429, "Urinishlar soni oshib ketdi. 15 daqiqadan keyin qayta urinib ko‘ring.",
                                headers={"Retry-After": "900"})
        if not token:
            raise HTTPException(401, "Login yoki parol xato.")
        user = accounts.session(token)
        if workspaces is not None:
            user = workspaces.identity(user)
            if user is None:
                accounts.logout(token)
                raise HTTPException(403, "Korxona ish maydoni faol emas.")
        accounts.logout(request.cookies.get(SESSION_COOKIE))
        response.headers["Cache-Control"] = "no-store"
        response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_TTL, httponly=True,
                            secure=cookie_secure(request), samesite="lax")
        return {"user": user, "workspaces": workspaces is not None}

    @router.post("/api/session/logout")
    def logout(request: Request, response: Response):
        if accounts is not None:
            accounts.logout(request.cookies.get(SESSION_COOKIE))
        response.delete_cookie(SESSION_COOKIE, secure=cookie_secure(request), httponly=True, samesite="lax")
        return {"ok": True}

    @router.post("/api/session/password")
    def password(body: PasswordUpdate, request: Request, response: Response):
        if accounts is None:
            raise HTTPException(404, "Hisoblar o‘chirilgan")
        try:
            changed = accounts.change_password(request.state.user["id"], body.current, body.password)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not changed:
            raise HTTPException(400, "Joriy parol xato.")
        response.delete_cookie(SESSION_COOKIE, secure=cookie_secure(request), httponly=True, samesite="lax")
        return {"ok": True}

    @router.get("/api/team")
    def team():
        return {"users": accounts.list()}

    @router.post("/api/team", status_code=201)
    def add_admin(body: Login):
        try:
            return accounts.create(body.username, body.password, role="admin")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @router.patch("/api/team/{user_id}")
    def edit_admin(user_id: int, body: AdminUpdate):
        try:
            user = accounts.edit_admin(user_id, body.active, body.password)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if user is None:
            raise HTTPException(404, "Admin topilmadi.")
        return user

    return router
