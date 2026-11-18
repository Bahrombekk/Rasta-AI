"""Konfiguratsiyani yuklash (config.yaml) va qurilma tanlash."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Loyiha ildizi (src/rasta/config.py -> ildiz)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"


class Config:
    """config.yaml ustidan nuqta bilan kirish imkonini beruvchi o'ram.

    Masalan: cfg.get("detector.conf", 0.2)
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        p = Path(path) if path else DEFAULT_CONFIG
        if not p.exists():
            raise FileNotFoundError(f"Konfiguratsiya topilmadi: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        node = self._data
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def override(self, items: list[str] | None) -> "Config":
        """["recognition.w_color=0.3", "device=cpu"] ko'rinishidagi o'zgartirishlar."""
        for it in items or []:
            key, _, raw = it.partition("=")
            try:
                val = yaml.safe_load(raw)
            except yaml.YAMLError:
                val = raw
            # ":memory:" kabi satrlar YAML'da lug'at bo'lib qoladi — satr sifatida olamiz
            if isinstance(val, dict) and not raw.strip().startswith("{"):
                val = raw.strip()
            self.set(key.strip(), val)
        return self

    def copy(self) -> "Config":
        import copy
        return Config(copy.deepcopy(self._data))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


def resolve_device(preference: str = "auto") -> str:
    """'auto' | 'cuda' | 'cpu' -> haqiqiy qurilma satri."""
    if preference and preference != "auto":
        return preference
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_path(p: str | os.PathLike) -> Path:
    """Nisbiy yo'lni loyiha ildiziga nisbatan hal qiladi."""
    path = Path(p)
    if path.is_absolute():
        return path
    return ROOT / path
