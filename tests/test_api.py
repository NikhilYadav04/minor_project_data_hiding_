import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.main import app


def _png_bytes() -> bytes:
    image = np.arange(512 * 512, dtype=np.uint8).reshape(512, 512)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_api_owner_and_recovery_flow():
    client = TestClient(app)
    response = client.post(
        "/api/owner/encode",
        files={"file": ("image.png", _png_bytes(), "image/png")},
        data={"seed": "21", "predictor": "med", "top": "300", "left": "300", "height": "60", "width": "40"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["shares"]) == 4
    for stage, indexes in ((1, [0, 1, 2]), (2, [0, 1, 2, 3]), (3, [0, 1, 2, 3])):
        recovered = client.post("/api/receiver/recover", json={"jobId": body["jobId"], "stage": stage, "shareIndexes": indexes})
        assert recovered.status_code == 200
        assert recovered.json()["exact"] is True


def test_api_automatic_recovery_selects_valid_threshold_subset():
    client = TestClient(app)
    response = client.post(
        "/api/owner/encode",
        files={"file": ("image.png", _png_bytes(), "image/png")},
        data={"seed": "22", "predictor": "med", "top": "300", "left": "300", "height": "60", "width": "40", "msa_mode": "auto_texture"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["msaMode"] == "auto_texture"
    assert body["msaRect"]["height"] == 60
    recovered = client.post("/api/receiver/recover-auto", json={"jobId": body["jobId"], "stage": 3, "shareIndexes": []})
    assert recovered.status_code == 200
    assert recovered.json()["exact"] is True
    assert recovered.json()["usedIndexes"] == [0, 1, 2]


def test_api_automatic_predictor_selection_reports_evidence():
    client = TestClient(app)
    response = client.post(
        "/api/owner/encode",
        files={"file": ("image.png", _png_bytes(), "image/png")},
        data={"seed": "24", "predictor": "auto", "top": "300", "left": "300", "height": "60", "width": "40"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["requestedPredictor"] == "auto"
    assert body["predictorSelection"] == "automatic"
    assert len(body["predictorCandidates"]) == 3
    assert body["predictor"] in {"bilinear", "edge_directed", "med"}


def test_api_quality_and_attack_simulation():
    client = TestClient(app)
    response = client.post(
        "/api/owner/encode",
        files={"file": ("image.png", _png_bytes(), "image/png")},
        data={"seed": "25", "predictor": "auto", "top": "300", "left": "300", "height": "60", "width": "40"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["quality"]["stage3"]["psnr"] == "∞"
    assert body["quality"]["stage3"]["ssim"] == 1.0
    assert body["quality"]["stage3"]["mae"] == 0.0
    attack = client.post("/api/receiver/attack", json={"jobId": body["jobId"], "attack": "one_pixel"})
    assert attack.status_code == 200
    assert attack.json()["detected"] is True
    assert attack.json()["exact"] is True
    assert attack.json()["validIndexes"] == [0, 2, 3]
    rejected = client.post("/api/receiver/attack", json={"jobId": body["jobId"], "attack": "two_missing"})
    assert rejected.status_code == 200
    assert rejected.json()["recovery"] == "rejected"


def test_api_optional_fidelity_backends_preserve_exact_recovery():
    client = TestClient(app)
    response = client.post(
        "/api/owner/encode",
        files={"file": ("image.png", _png_bytes(), "image/png")},
        data={
            "seed": "26",
            "predictor": "rcan_style",
            "inpainting": "contextual",
            "top": "300",
            "left": "300",
            "height": "60",
            "width": "40",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["inpaintingMethod"] == "contextual"
    assert body["stats"]["predictor_backend"] == "rcan_style_fixed_point"
    recovered = client.post("/api/receiver/recover-auto", json={"jobId": body["jobId"], "stage": 3, "shareIndexes": []})
    assert recovered.status_code == 200
    assert recovered.json()["exact"] is True
