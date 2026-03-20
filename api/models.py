# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import MODEL_ROOT, MODEL_NAMES, DEFAULT_MAX_LENGTH, USE_MODELS
from utils import log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SingleModel:
    def __init__(self, name: str, src_dir: Path, max_length: int = DEFAULT_MAX_LENGTH):
        self.name = name
        self.src = str(src_dir)
        self.max_length = max_length
        self.available = False
        self.error = None
        self.pos_idx = 1
        self.tok = None
        self.mdl = None

    def _detect_pos_index(self):
        id2lab = getattr(self.mdl.config, "id2label", None)
        if isinstance(id2lab, dict):
            try:
                norm = {int(k): str(v).lower() for k, v in id2lab.items()}
            except Exception:
                norm = {int(k): str(v).lower() for k, v in id2lab.items()}
            for k, v in norm.items():
                if v in ("positive", "positif", "presence", "pos", "1", "true"):
                    return int(k)
            if norm.get(0, "") in ("negative", "negatif", "absence", "neg", "0", "false"):
                return 1
        return 1

    def load(self):
        try:
            self.tok = AutoTokenizer.from_pretrained(self.src, use_fast=True)
            self.mdl = AutoModelForSequenceClassification.from_pretrained(self.src, num_labels=2).to(DEVICE).eval()
            self.pos_idx = self._detect_pos_index()
            self.available = True
            self.error = None
            log(f"[MODEL] '{self.name}' chargé ({self.src}) pos_idx={self.pos_idx}")
        except Exception as e:
            self.available = False
            self.error = str(e)
            log(f"[MODEL] ERREUR chargement '{self.name}': {e}")

    @torch.no_grad()
    def predict_probs(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not self.available:
            raise RuntimeError(f"Model '{self.name}' not available: {self.error}")
        pres = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(
                texts[i:i + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            logits = self.mdl(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, self.pos_idx].detach().cpu().numpy()
            pres.append(p)
        return np.concatenate(pres, axis=0) if pres else np.array([], dtype=float)

# registre global
MODELS: Dict[str, SingleModel] = {
    name: SingleModel(name, MODEL_ROOT / name, DEFAULT_MAX_LENGTH) for name in MODEL_NAMES
}
_loading = {name: {"in_progress": False, "error": None} for name in MODEL_NAMES}

def _load_one(name: str):
    _loading[name]["in_progress"] = True
    _loading[name]["error"] = None
    try:
        MODELS[name].load()
        if not MODELS[name].available:
            _loading[name]["error"] = MODELS[name].error or "unknown"
    except Exception as e:
        _loading[name]["error"] = str(e)
    finally:
        _loading[name]["in_progress"] = False

def load_all_models_blocking():
    if not USE_MODELS:
        return
    for name in MODEL_NAMES:
        _load_one(name)

def models_state():
    states = []
    for name, m in MODELS.items():
        states.append({
            "name": name,
            "src": m.src,
            "available": bool(m.available),
            "pos_idx": m.pos_idx,
            "max_length": m.max_length,
            "loading": bool(_loading[name]["in_progress"]),
            "load_error": _loading[name]["error"] or m.error,
        })
    return states

def any_loading():
    return any(s["loading"] for s in models_state())

def any_unavailable():
    return any((not s["available"]) for s in models_state())

def log_models_overview():
    for n, m in MODELS.items():
        log(f"[BOOT] Model '{n}' -> {getattr(m, 'src', 'N/A')}")
