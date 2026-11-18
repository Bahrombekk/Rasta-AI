"""src/ ni import yo'liga qo'shish + Windows konsolida UTF-8 ni majburlash."""
import sys
from pathlib import Path

# Windows konsoli (cp1251) '✓' va emoji belgilarini chiqara olmaydi -> UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
for directory in ("src", "backend", "ai"):
    source = str(ROOT / directory)
    if source not in sys.path:
        sys.path.insert(0, source)
