"""Rasta veb-ilovasini ishga tushirish.

  python scripts/serve.py                  # http://127.0.0.1:8000  (admin: /admin)
  python scripts/serve.py --host 0.0.0.0 --port 8080
  set RASTA_ADMIN_PASSWORD=parol           # admin amallari uchun parol (ixtiyoriy)
"""
import argparse
import os

import _bootstrap  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    import uvicorn

    from rasta.config import Config
    from rasta.web import app as web

    cfg = Config.load(args.config).override(args.set)
    public = args.host not in ("127.0.0.1", "localhost", "::1")
    if public and not os.environ.get("RASTA_ADMIN_PASSWORD"):
        print("[!] Server tarmoqqa ochiq, RASTA_ADMIN_PASSWORD yo'q — admin amallari o'chirildi")
    application = web.create_app(cfg, public=public)
    print("[+] Modellar yuklanmoqda (birinchi marta ~10-20 s)...")
    web.get_pipe()
    print(f"[✓] Tayyor: http://{args.host}:{args.port}   admin: http://{args.host}:{args.port}/admin")
    uvicorn.run(application, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
