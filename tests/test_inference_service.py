from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rasta import inference
from rasta.ai_service import create_ai_app
from rasta.config import Config
from rasta.types import Box


def test_remote_inference_validates_signature_and_splits_batches(monkeypatch):
    cfg = Config({"embedder": {"model": "test"}})
    token = "a" * 40
    calls = []

    def embed(images):
        calls.append(len(images))
        return np.array([[image[0, 0, 0], image[0, 0, 1]] for image in images], dtype=np.float32)

    worker = create_ai_app(cfg, token, detector=SimpleNamespace(detect=lambda image: [Box(0, 0, 3, 2, .8)]),
                           embedder=SimpleNamespace(dim=2, embed=embed))
    assert TestClient(worker).get("/health").status_code == 401
    monkeypatch.setattr(inference.requests, "Session", lambda: TestClient(worker, base_url="http://worker"))
    remote = inference.RemoteInference(cfg, "http://worker", token)
    image = np.full((4, 5, 3), [19, 199, 217], dtype=np.uint8)
    assert remote.detect(image) == [Box(0, 0, 3, 2, .8)]
    np.testing.assert_array_equal(remote.embed([image] * 33), np.tile([19, 199], (33, 1)))
    assert calls == [32, 1]
    with pytest.raises(inference.InferenceUnavailable):
        inference.RemoteInference(Config({}), "http://worker", token)
