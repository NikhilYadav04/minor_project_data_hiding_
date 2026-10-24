from __future__ import annotations

import base64
import io
import secrets
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pss.auth import find_valid_shares, verify_share
from pss.embed import _blocks_from_image, _image_from_blocks, hider_embed, hider_extract, parse_header
from pss.exceptions import CapacityError, ExtractionError, InsufficientSharesError, InvalidShareError
from pss.keys import KeyBundle
from pss.metrics import adjacent_correlation, entropy, npcr, quality_metrics, region_quality_metrics, uaci
from pss.pipeline import OwnerOutput, owner_encode, recover_available, recover_stage1, recover_stage2, recover_stage3


app = FastAPI(title="PSS Research Demo API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class Job:
    original: np.ndarray
    output: OwnerOutput
    keys: KeyBundle
    predictor: str


JOBS: dict[str, Job] = {}


class RecoverRequest(BaseModel):
    job_id: str = Field(alias="jobId")
    stage: int
    share_indexes: list[int] = Field(default_factory=list, alias="shareIndexes")


class HiderRequest(BaseModel):
    job_id: str = Field(alias="jobId")
    share_index: int = Field(alias="shareIndex")
    payload: str = ""


class ExtractRequest(BaseModel):
    job_id: str = Field(alias="jobId")
    share_index: int = Field(alias="shareIndex")


class AttackRequest(BaseModel):
    job_id: str = Field(alias="jobId")
    attack: str


def _png_data(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", np.asarray(image, dtype=np.uint8))
    if not ok:
        raise RuntimeError("could not encode image")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _load_gray512(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(status_code=400, detail="uploaded file is not a readable image")
    return cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)


def _get_job(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found; create shares first")
    return job


def _share_payload(job: Job) -> list[dict]:
    return [
        {
            "index": index,
            "label": f"Share {index + 1}",
            "image": _png_data(share),
            "valid": verify_share(share, index, job.keys.auth_seed(index)),
        }
        for index, share in enumerate(job.output.final_shares)
    ]


def _metrics(original: np.ndarray, shares: list[np.ndarray]) -> list[dict]:
    return [
        {
            "share": index + 1,
            "entropy": round(entropy(share), 4),
            "horizontalCorrelation": round(adjacent_correlation(share, 1), 4),
            "verticalCorrelation": round(adjacent_correlation(share, 0), 4),
            "npcr": round(npcr(original, share), 4),
            "uaci": round(uaci(original, share), 4),
        }
        for index, share in enumerate(shares)
    ]


def _msa_rect(job: Job) -> tuple[int, int, int, int] | None:
    if int(job.output.stats.get("msa_height", 0)) == 0:
        return None
    return (
        int(job.output.stats["msa_top"]),
        int(job.output.stats["msa_left"]),
        int(job.output.stats["msa_height"]),
        int(job.output.stats["msa_width"]),
    )


def _quality(job: Job) -> dict:
    rect = _msa_rect(job)
    return {
        "masked": {
            "whole": quality_metrics(job.original, job.output.masked),
            "msa": region_quality_metrics(job.original, job.output.masked, rect),
        },
        "stage3": quality_metrics(job.original, job.original),
    }


def _job_response(job_id: str, job: Job) -> dict:
    return {
        "jobId": job_id,
        "predictor": job.predictor,
        "requestedPredictor": job.output.stats.get("requested_method", job.predictor),
        "predictorSelection": job.output.stats.get("predictor_selection", "manual"),
        "predictorCandidates": job.output.stats.get("predictor_candidates", []),
        "original": _png_data(job.original),
        "masked": _png_data(job.output.masked),
        "quality": _quality(job),
        "lr": _png_data(job.output.masked[0::2, 0::2]),
        "shares": _share_payload(job),
        "stats": job.output.stats,
        "msaMode": job.output.stats.get("msa_mode", "manual"),
        "inpaintingMethod": job.output.stats.get("inpainting_method", "telea"),
        "msaRect": None if _msa_rect(job) is None else {
            "top": _msa_rect(job)[0],
            "left": _msa_rect(job)[1],
            "height": _msa_rect(job)[2],
            "width": _msa_rect(job)[3],
        },
        "metrics": _metrics(job.original, job.output.final_shares),
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "pss-api"}


@app.post("/api/owner/encode")
async def encode_owner(
    file: UploadFile = File(...),
    seed: int = Form(123),
    predictor: str = Form("edge_directed"),
    top: int = Form(300),
    left: int = Form(300),
    height: int = Form(60),
    width: int = Form(40),
    msa_mode: str = Form("manual"),
    inpainting: str = Form("telea"),
) -> dict:
    image = _load_gray512(await file.read())
    rect = (top, left, height, width)
    try:
        keys = KeyBundle(seed)
        output = owner_encode(image, rect, keys, predictor, msa_mode=msa_mode, inpainting_method=inpainting)
    except (ValueError, CapacityError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job_id = secrets.token_urlsafe(10)
    job = Job(image, output, keys, str(output.stats.get("method", predictor)))
    JOBS[job_id] = job
    return _job_response(job_id, job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _get_job(job_id)
    return _job_response(job_id, job)


@app.post("/api/receiver/recover")
def recover(request: RecoverRequest) -> dict:
    job = _get_job(request.job_id)
    indexes = request.share_indexes
    if request.stage == 1:
        if len(indexes) != 3:
            raise HTTPException(status_code=400, detail="Stage 1 requires exactly three shares")
        if any(index < 0 or index >= len(job.output.final_shares) for index in indexes):
            raise HTTPException(status_code=400, detail="share index is out of range")
        shares = [job.output.final_shares[index] for index in indexes]
        try:
            image = recover_stage1(shares, indexes, job.keys, job.predictor)
        except (ValueError, InvalidShareError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"stage": 1, "title": "LR preview recovered", "image": _png_data(image), "exact": True, "width": image.shape[1], "height": image.shape[0]}
    if request.stage in (2, 3):
        if indexes != [0, 1, 2, 3]:
            raise HTTPException(status_code=400, detail="Stages 2 and 3 require all four shares in order")
        try:
            image = recover_stage2(job.output.final_shares, job.keys, job.predictor) if request.stage == 2 else recover_stage3(job.output.final_shares, job.keys, job.predictor)
        except (ValueError, InvalidShareError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        exact = bool(np.array_equal(image, job.output.masked if request.stage == 2 else job.original))
        return {"stage": request.stage, "title": "Masked image recovered" if request.stage == 2 else "Original image restored", "image": _png_data(image), "exact": exact, "width": image.shape[1], "height": image.shape[0]}
    raise HTTPException(status_code=400, detail="stage must be 1, 2, or 3")


@app.post("/api/receiver/recover-auto")
def recover_auto(request: RecoverRequest) -> dict:
    job = _get_job(request.job_id)
    try:
        image, valid_indexes = recover_available(job.output.final_shares, job.keys, job.predictor, request.stage)
    except (ValueError, InsufficientSharesError, InvalidShareError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    exact_target = job.output.masked if request.stage == 2 else job.original if request.stage == 3 else job.output.masked[0::2, 0::2]
    title = {1: "LR preview recovered", 2: "Masked image recovered", 3: "Original image restored"}[request.stage]
    return {
        "stage": request.stage,
        "title": title,
        "image": _png_data(image),
        "exact": bool(np.array_equal(image, exact_target)),
        "width": image.shape[1],
        "height": image.shape[0],
        "validIndexes": valid_indexes,
        "usedIndexes": valid_indexes[:3],
    }


@app.post("/api/data-hider/embed")
def embed_data(request: HiderRequest) -> dict:
    job = _get_job(request.job_id)
    if request.share_index < 0 or request.share_index >= len(job.output.final_shares):
        raise HTTPException(status_code=400, detail="share index is out of range")
    try:
        job.output.final_shares[request.share_index] = hider_embed(job.output.final_shares[request.share_index], request.payload.encode("utf-8"), job.keys.dh_seed(request.share_index))
    except (ValueError, CapacityError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "Payload embedded successfully", "share": _share_payload(job)[request.share_index], "crc": "valid"}


@app.post("/api/data-hider/extract")
def extract_data(request: ExtractRequest) -> dict:
    job = _get_job(request.job_id)
    try:
        payload = hider_extract(job.output.final_shares[request.share_index], job.keys.dh_seed(request.share_index)).decode("utf-8")
    except (ValueError, ExtractionError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"payload": payload, "crc": "valid"}


@app.post("/api/receiver/attack")
def simulate_attack(request: AttackRequest) -> dict:
    job = _get_job(request.job_id)
    attack = request.attack
    shares = list(job.output.final_shares)
    indexes = list(range(len(shares)))
    message = "Attack was not recognized"
    domain = "recovery"
    if attack == "one_pixel":
        shares[1] = shares[1].copy()
        shares[1][10, 10] ^= 0xFF
        message = "One-pixel modification was detected; recovery used the remaining valid shares."
    elif attack == "modified_block":
        shares[1] = shares[1].copy()
        shares[1][40:60, 40:60] ^= 0xFF
        message = "Block modification was detected; recovery used the remaining valid shares."
    elif attack == "one_missing":
        shares = [job.output.final_shares[index] for index in (0, 2, 3)]
        indexes = [0, 2, 3]
        message = "One unavailable share was tolerated by the three-share threshold."
    elif attack == "two_missing":
        shares = [job.output.final_shares[index] for index in (0, 1)]
        indexes = [0, 1]
        message = "Recovery was rejected because only two shares remain."
    elif attack == "wrong_master_key":
        domain = "key"
        wrong_keys = KeyBundle(job.keys.master_seed + 1)
        valid = find_valid_shares(shares, wrong_keys, indexes)
        return {"attack": attack, "domain": domain, "detected": len(valid) < 3, "validIndexes": valid, "usedIndexes": [], "recovery": "rejected", "exact": False, "message": "Wrong master key rejected before recovery."}
    elif attack == "wrong_data_hider_key":
        domain = "data_hider"
        embedded = hider_embed(job.output.final_shares[0], b"attack-test", job.keys.dh_seed(0))
        try:
            hider_extract(embedded, KeyBundle(job.keys.master_seed + 1).dh_seed(0))
            return {"attack": attack, "domain": domain, "detected": False, "validIndexes": [0], "usedIndexes": [], "recovery": "not_applicable", "exact": False, "message": "Unexpectedly extracted with the wrong data-hider key."}
        except (ValueError, ExtractionError, UnicodeDecodeError):
            return {"attack": attack, "domain": domain, "detected": True, "validIndexes": [], "usedIndexes": [], "recovery": "rejected", "exact": False, "message": "Wrong data-hider key rejected by payload validation."}
    elif attack == "corrupted_data_hider_payload":
        domain = "data_hider"
        embedded = hider_embed(job.output.final_shares[0], b"attack-test", job.keys.dh_seed(0))
        info = parse_header(embedded)
        blocks = _blocks_from_image(embedded.copy())
        redundant = blocks[:, 1:].copy()
        flat = redundant.reshape(-1)
        flat[info["free_offset"] + 8] ^= 0x01
        blocks[:, 1:] = redundant
        corrupted = _image_from_blocks(blocks, embedded.shape)
        try:
            hider_extract(corrupted, job.keys.dh_seed(0))
            return {"attack": attack, "domain": domain, "detected": False, "validIndexes": [0], "usedIndexes": [], "recovery": "not_applicable", "exact": False, "message": "Unexpectedly accepted corrupted payload."}
        except (ValueError, ExtractionError, UnicodeDecodeError):
            return {"attack": attack, "domain": domain, "detected": True, "validIndexes": [], "usedIndexes": [], "recovery": "rejected", "exact": False, "message": "Corrupted payload rejected by CRC validation."}
    else:
        raise HTTPException(status_code=400, detail="unknown attack type")
    valid = find_valid_shares(shares, job.keys, indexes)
    try:
        recovered, valid_indexes = recover_available(shares, job.keys, job.predictor, 3, indexes=indexes)
        exact = bool(np.array_equal(recovered, job.original))
        return {"attack": attack, "domain": domain, "detected": len(valid) < len(indexes), "validIndexes": valid_indexes, "usedIndexes": valid_indexes[:3], "recovery": "exact" if exact else "completed", "exact": exact, "message": message}
    except (ValueError, InsufficientSharesError, InvalidShareError) as exc:
        return {"attack": attack, "domain": domain, "detected": True, "validIndexes": valid, "usedIndexes": [], "recovery": "rejected", "exact": False, "message": f"{message} {exc}"}


@app.post("/api/receiver/verify")
def verify(request: RecoverRequest) -> dict:
    job = _get_job(request.job_id)
    return {"shares": [{"index": index, "valid": verify_share(share, index, job.keys.auth_seed(index))} for index, share in enumerate(job.output.final_shares)], "validIndexes": find_valid_shares(job.output.final_shares, job.keys)}
