"""
API d’inférence MRAC-FND — serveur central pour l’extension Chrome.
Déployé depuis : https://github.com/AdamTlili841/ServeurMRAC
"""

from __future__ import annotations

import io
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import BertTokenizer, ViTImageProcessor

from multimodal_model import load_from_checkpoint

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(os.environ.get("CHECKPOINT_PATH", str(ROOT / "best_model.pt")))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRELOAD_MODEL = os.environ.get("PRELOAD_MODEL", "true").lower() in (
    "1",
    "true",
    "yes",
)

_model = None
_tokenizer: BertTokenizer | None = None
_processor: ViTImageProcessor | None = None
_model_error: str | None = None


def _load_stack() -> tuple[torch.nn.Module, BertTokenizer, ViTImageProcessor]:
    global _model, _tokenizer, _processor, _model_error
    if _model is None:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"Checkpoint introuvable : {CHECKPOINT}. "
                "Ajoutez best_model.pt à la racine du dépôt (voir README)."
            )
        model, _cfg = load_from_checkpoint(str(CHECKPOINT), DEVICE)
        from multimodal_model import BERT_ID, VIT_ID

        _tokenizer = BertTokenizer.from_pretrained(BERT_ID)
        _processor = ViTImageProcessor.from_pretrained(VIT_ID)
        _model = model
        _model_error = None
    assert _tokenizer is not None and _processor is not None and _model is not None
    return _model, _tokenizer, _processor


def _preload_model() -> None:
    global _model_error
    try:
        print(f"MRAC-FND: chargement du modèle sur {DEVICE}…", flush=True)
        _load_stack()
        print("MRAC-FND: modèle prêt.", flush=True)
    except Exception as exc:
        _model_error = str(exc)
        traceback.print_exc()
        print(f"MRAC-FND: échec préchargement — {exc}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if PRELOAD_MODEL:
        _preload_model()
    yield


app = FastAPI(
    title="ServeurMRAC",
    description="API MRAC-FND — détection multimodale fake news (texte + image)",
    version="1.0.1",
    lifespan=lifespan,
)

_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "ServeurMRAC", "docs": "/docs", "health": "/health", "ready": "/ready"}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "checkpoint": CHECKPOINT.name,
        "device": str(DEVICE),
        "model_loaded": _model is not None,
        "model_error": _model_error,
    }


@app.get("/ready")
def ready():
    if _model is not None:
        return {"status": "ready", "device": str(DEVICE)}
    if _model_error:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": _model_error},
        )
    return JSONResponse(
        status_code=503,
        content={"status": "loading", "detail": "Modèle en cours de chargement."},
    )


def _blank_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(245, 245, 245))


@app.post("/predict")
async def predict(
    text: str = Form(...),
    image: UploadFile | None = File(None),
) -> dict:
    if _model_error and _model is None:
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    f"Modèle non chargé : {_model_error}. "
                    "Sur Render, utilisez au minimum le plan Standard (2 Go RAM)."
                )
            },
        )

    model, tokenizer, processor = _load_stack()

    pil = _blank_image()
    if image is not None:
        raw = await image.read()
        if raw:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")

    tv = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    pv = processor(images=pil, return_tensors="pt")

    input_ids = tv["input_ids"].to(DEVICE)
    attention_mask = tv["attention_mask"].to(DEVICE)
    pixel_values = pv["pixel_values"].to(DEVICE)

    with torch.inference_mode():
        logits = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        probs = torch.softmax(logits, dim=-1)[0]
        fake_prob = float(probs[1]) if probs.numel() > 1 else float(probs[0])

    verdict = "fake" if fake_prob >= 0.5 else "real"

    return {
        "verdict": verdict,
        "confidence_fake": fake_prob,
        "probs": [float(probs[0]), float(probs[1])] if probs.numel() > 1 else [float(probs[0])],
        "demo": False,
        "explanation": "Prédiction du modèle MRAC-FND (serveur central).",
    }
