"""Run the private Rasta inference service."""
import argparse
import os

import uvicorn

from rasta_ai.service import create_ai_app
from rasta.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    cfg = Config.load(args.config).override(args.set)
    app = create_ai_app(cfg, os.environ.get("RASTA_AI_TOKEN", ""))
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
