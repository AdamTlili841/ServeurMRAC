"""
API d’inférence MRAC-FND — serveur central pour l’extension Chrome.
Déployé depuis : https://github.com/AdamTlili841/ServeurMRAC
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import BertTokenizer, ViTImageProcessor

from multimodal_model import load_from_checkpoint

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(os.environ.get("CHECKPOINT_PATH", str(ROOT / "best_model.pt")))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="ServeurMRAC",
    description="API MRAC-FND — détection multimodale fake news (texte + image)",
    version="1.0.0",
)

_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_tokenizer: BertTokenizer | None = None
_processor: ViTImageProcessor | None = None


def _load_stack() -> tuple[torch.nn.Module, BertTokenizer, ViTImageProcessor]:
    global _model, _tokenizer, _processor
    if _model is None:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"Checkpoint introuvable : {CHECKPOINT}. "
                "Ajoutez best_model.pt à la racine du dépôt (voir README)."
            )
        model, _cfg = load_from_checkpoint(str(CHECKPOINT), DEVICE)
        _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        _model = model
    assert _tokenizer is not None and _processor is not None and _model is not None
    return _model, _tokenizer, _processor


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "ServeurMRAC", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "checkpoint": str(CHECKPOINT.name)}


def _blank_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(245, 245, 245))


@app.post("/predict")
async def predict(
    text: str = Form(...),
    image: UploadFile | None = File(None),
) -> dict:
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

    with torch.no_grad():
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
