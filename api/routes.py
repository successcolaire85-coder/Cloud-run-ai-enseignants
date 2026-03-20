# -*- coding: utf-8 -*-
from flask import Blueprint, jsonify, request
from typing import List, Dict, Any

from config import DEFAULT_THRESHOLD, MODEL_NAMES, USE_MODELS
from models import MODELS, models_state, any_loading, any_unavailable
from utils import form_to_text, concat_row_for_csv

api_bp = Blueprint("api", __name__)

# ---------- Health & Warmup ----------

@api_bp.get("/health")
def health():
    mstate = models_state()
    overall = "ok"
    if any_loading():
        overall = "initializing"
    elif any_unavailable() and USE_MODELS:
        overall = "error"
    return jsonify({
        "status": overall,
        "use_models": USE_MODELS,
        "default_threshold": DEFAULT_THRESHOLD,
        "models": mstate
    })

@api_bp.post("/warmup")
def warmup():
    # ici on ne relance pas async (chargement déjà bloquant au boot)
    return jsonify({"status": "noop (models loaded at boot)", "models": MODEL_NAMES}), 200

# ---------- Unitaire ----------

@api_bp.post("/predict_label")
def predict_label():
    data = request.get_json(force=True) or {}
    thr = float(data.get("threshold", DEFAULT_THRESHOLD))
    model_name = str(data.get("model", MODEL_NAMES[0])).strip().lower()
    if model_name not in MODELS:
        return jsonify({"error": f"unknown model '{model_name}'", "choices": MODEL_NAMES}), 400

    if "form" in data:
        txt = form_to_text(data["form"])
    else:
        txt = str(data.get("text", "")).strip()
    if not txt:
        return jsonify({"error": "no text/form provided"}), 400

    if USE_MODELS:
        m = MODELS[model_name]
        if not m.available:
            return jsonify({"error": f"model '{model_name}' not available", "load_error": m.error}), 503
        p = float(m.predict_probs([txt])[0])
    else:
        p = 0.5

    y = 1 if p >= thr else 0
    return jsonify({"label": y, "prob": p, "threshold": thr, "model": model_name, "used_text": txt})

# ---------- Batch simple (liste de textes) ----------

@api_bp.post("/predict_batch")
def predict_batch():
    data = request.get_json(force=True) or {}
    thr = float(data.get("threshold", DEFAULT_THRESHOLD))
    model_name = str(data.get("model", MODEL_NAMES[0])).strip().lower()
    if model_name not in MODELS:
        return jsonify({"error": f"unknown model '{model_name}'", "choices": MODEL_NAMES}), 400

    texts = [str(t).strip() for t in (data.get("texts") or []) if str(t).strip() != ""]
    if not texts:
        return jsonify({"error": "no texts provided"}), 400

    if USE_MODELS:
        m = MODELS[model_name]
        if not m.available:
            return jsonify({"error": f"model '{model_name}' not available", "load_error": m.error}), 503
        probs = m.predict_probs(texts).tolist()
    else:
        probs = [0.5] * len(texts)

    preds = [1 if float(p) >= thr else 0 for p in probs]
    return jsonify({
        "threshold": thr,
        "model": model_name,
        "per_model": {model_name: {"probs": probs}},
        "preds": preds
    })

# ---------- Formulaire -> texte ----------

@api_bp.post("/predict_form")
def predict_form():
    data = request.get_json(force=True) or {}
    if "form" not in data or not isinstance(data["form"], dict):
        return jsonify({"error": "form (dict) is required"}), 400
    thr = float(data.get("threshold", DEFAULT_THRESHOLD))
    model_name = str(data.get("model", MODEL_NAMES[0])).strip().lower()
    if model_name not in MODELS:
        return jsonify({"error": f"unknown model '{model_name}'", "choices": MODEL_NAMES}), 400
    txt = form_to_text(data["form"])

    if USE_MODELS:
        m = MODELS[model_name]
        if not m.available:
            return jsonify({"error": f"model '{model_name}' not available", "load_error": m.error}), 503
        p = float(m.predict_probs([txt])[0])
    else:
        p = 0.5
    y = 1 if p >= thr else 0
    return jsonify({"label": y, "prob": p, "threshold": thr, "model": model_name, "used_text": txt})

# ---------- Multi-modèles ----------

@api_bp.post("/predict_multi")
def predict_multi():
    data = request.get_json(force=True) or {}
    thr = float(data.get("threshold", DEFAULT_THRESHOLD))

    req_models = data.get("models") or MODEL_NAMES
    req_models = [str(m).strip().lower() for m in req_models if str(m).strip().lower() in MODEL_NAMES]
    if not req_models:
        return jsonify({"error": "no valid models specified", "choices": MODEL_NAMES}), 400

    used_text = None
    texts = None
    if "texts" in data and isinstance(data["texts"], list):
        texts = [str(t).strip() for t in data["texts"] if str(t).strip() != ""]
        if not texts:
            return jsonify({"error": "no texts provided"}), 400
    else:
        if "form" in data and isinstance(data["form"], dict):
            used_text = form_to_text(data["form"])
        else:
            used_text = str(data.get("text", "")).strip()
        if not used_text:
            return jsonify({"error": "no text/form provided"}), 400

    if texts is not None:
        per_model = {}
        for name in req_models:
            m = MODELS[name]
            if USE_MODELS:
                if not m.available:
                    per_model[name] = {"error": f"model '{name}' not available", "load_error": m.error}
                    continue
                probs = m.predict_probs(texts).tolist()
            else:
                probs = [0.5] * len(texts)
            preds = [1 if float(p) >= thr else 0 for p in probs]
            per_model[name] = {"probs": probs, "preds": preds}
        return jsonify({"threshold": thr, "models": req_models, "per_model": per_model})
    else:
        per_model = {}
        for name in req_models:
            m = MODELS[name]
            if USE_MODELS:
                if not m.available:
                    per_model[name] = {"error": f"model '{name}' not available", "load_error": m.error}
                    continue
                p = float(m.predict_probs([used_text])[0])
            else:
                p = 0.5
            y = 1 if p >= thr else 0
            per_model[name] = {"prob": p, "label": y}
        return jsonify({"threshold": thr, "models": req_models, "per_model": per_model, "used_text": used_text})

# ---------- Helpers CSV (texte direct) ----------

@api_bp.post("/predict_csv_text")
def predict_csv_text():
    data = request.get_json(force=True) or {}
    thr = float(data.get("threshold", DEFAULT_THRESHOLD))
    model_name = str(data.get("model", MODEL_NAMES[0])).strip().lower()
    if model_name not in MODELS:
        return jsonify({"error": f"unknown model '{model_name}'", "choices": MODEL_NAMES}), 400

    rows = data.get("rows") or []
    texts = []
    for r in rows:
        t = "" if r is None else str(r.get("text", "")).strip()
        if t:
            texts.append(t)
    if not texts:
        return jsonify({"error": "no valid 'text' rows"}), 400

    if USE_MODELS:
        m = MODELS[model_name]
        if not m.available:
            return jsonify({"error": f"model '{model_name}' not available", "load_error": m.error}), 503
        probs = m.predict_probs(texts).tolist()
    else:
        probs = [0.5] * len(texts)
    preds = [1 if float(p) >= thr else 0 for p in probs]
    return jsonify({"model": model_name, "threshold": thr, "probs": probs, "preds": preds})

# ---------- Helpers CSV (concat Q24→Q36 [+hours]) ----------

@api_bp.post("/predict_csv_concat")
def predict_csv_concat():
    data = request.get_json(force=True) or {}
    thr = float(data.get("threshold", DEFAULT_THRESHOLD))
    model_name = str(data.get("model", MODEL_NAMES[0])).strip().lower()
    include_hours = bool(data.get("include_hours", True))
    if model_name not in MODELS:
        return jsonify({"error": f"unknown model '{model_name}'", "choices": MODEL_NAMES}), 400

    rows = data.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "rows (list of dicts) required"}), 400

    texts = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        texts.append(concat_row_for_csv(r, include_hours=include_hours))
    texts = [t for t in texts if t.strip()]
    if not texts:
        return jsonify({"error": "no valid text built from rows"}), 400

    if USE_MODELS:
        m = MODELS[model_name]
        if not m.available:
            return jsonify({"error": f"model '{model_name}' not available", "load_error": m.error}), 503
        probs = m.predict_probs(texts).tolist()
    else:
        probs = [0.5] * len(texts)
    preds = [1 if float(p) >= thr else 0 for p in probs]
    return jsonify({"model": model_name, "threshold": thr, "probs": probs, "preds": preds, "used_texts": texts})
