"""Run the private Rasta label-reading service."""
import argparse
import os

import uvicorn

from rasta_ocr.service import create_ocr_app


def selftest():
    """Which device the models run on and how long one read takes (python -m rasta_ocr --selftest)."""
    import time

    import numpy as np
    from PIL import Image, ImageDraw

    from rasta_ocr.service import PaddleReader

    image = Image.new("RGB", (900, 1400), "white")
    draw = ImageDraw.Draw(image)
    for row, text in enumerate(("Ays Tea", "Shaftoli", "1,25 L", "qora choy")):
        draw.text((120, 160 + row * 260), text, fill="black", font_size=120)
    pixels = np.array(image)
    reader = PaddleReader()
    reader.read(pixels)                               # warm-up
    started = time.perf_counter()
    rows = [reader.read(pixels) for _ in range(5)][-1]
    print(f"device={reader.device} | {(time.perf_counter() - started) / 5 * 1000:.0f} ms per crop | "
          f"read: {[row[0] for row in rows]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    app = create_ocr_app(os.environ.get("RASTA_AI_TOKEN", ""))
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
