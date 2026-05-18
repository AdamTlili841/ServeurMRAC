"""
Architecture alignée avec le checkpoint FineFake multimodal :
ViT-base + BERT-base + co-attention i→t et t→i, puis fusion et classification binaire.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel


class FakeNewsDualEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.txt_encoder = BertModel.from_pretrained("bert-base-uncased")

        hv = self.vit.config.hidden_size
        ht = self.txt_encoder.config.hidden_size
        self.img_proj = nn.Linear(hv, hidden_dim)
        self.txt_proj = nn.Linear(ht, hidden_dim)

        dim = hidden_dim

        self.q_i2t = nn.Linear(dim, dim)
        self.k_i2t = nn.Linear(dim, dim)
        self.v_i2t = nn.Linear(dim, dim)
        self.out_i2t = nn.Linear(dim, dim)

        self.q_t2i = nn.Linear(dim, dim)
        self.k_t2i = nn.Linear(dim, dim)
        self.v_t2i = nn.Linear(dim, dim)
        self.out_t2i = nn.Linear(dim, dim)

        self.coatt_fusion = nn.Sequential(nn.Linear(dim * 2, dim))
        self.fusion_proj = nn.Sequential(nn.Linear(dim * 3, dim))
        self.classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 2),
        )

        self._scale = math.sqrt(float(dim))

    def _cross_attn(
        self,
        q_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        ql: nn.Linear,
        kl: nn.Linear,
        vl: nn.Linear,
        ol: nn.Linear,
        key_padding_mask_kv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = ql(q_tokens)
        k = kl(kv_tokens)
        v = vl(v_tokens)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self._scale
        if key_padding_mask_kv is not None:
            attn = attn.masked_fill(key_padding_mask_kv.unsqueeze(1), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = ol(out)
        pooled = out.mean(dim=1)
        return out, pooled

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        vit_out = self.vit(pixel_values=pixel_values)
        txt_out = self.txt_encoder(input_ids=input_ids, attention_mask=attention_mask)

        img_h = vit_out.last_hidden_state
        txt_h = txt_out.last_hidden_state

        img_seq = self.img_proj(img_h)
        txt_seq = self.txt_proj(txt_h)

        txt_key_padding_mask = attention_mask == 0
        attn_i2t, _pool_i = self._cross_attn(
            img_seq,
            txt_seq,
            txt_seq,
            self.q_i2t,
            self.k_i2t,
            self.v_i2t,
            self.out_i2t,
            key_padding_mask_kv=txt_key_padding_mask,
        )
        attn_t2i, _pool_t = self._cross_attn(
            txt_seq,
            img_seq,
            img_seq,
            self.q_t2i,
            self.k_t2i,
            self.v_t2i,
            self.out_t2i,
            key_padding_mask_kv=None,
        )

        pooled_i2t = attn_i2t.mean(dim=1)
        pooled_t2i = attn_t2i.mean(dim=1)
        fused_coatt = self.coatt_fusion(torch.cat([pooled_i2t, pooled_t2i], dim=-1))

        img_cls = vit_out.pooler_output if vit_out.pooler_output is not None else img_h[:, 0, :]
        txt_cls = txt_out.pooler_output
        assert txt_cls is not None

        z_img = self.img_proj(img_cls)
        z_txt = self.txt_proj(txt_cls)
        z = torch.cat([z_txt, z_img, fused_coatt], dim=-1)
        fused = self.fusion_proj(z)
        return self.classifier(fused)


def build_model(cfg: dict[str, Any]) -> FakeNewsDualEncoder:
    hd = int(cfg.get("hidden_dim", 512))
    return FakeNewsDualEncoder(hidden_dim=hd)


def load_from_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
) -> tuple[FakeNewsDualEncoder, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
    model = build_model(cfg)
    state = payload["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, cfg
