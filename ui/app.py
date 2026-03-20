# app.py — UI Streamlit pour Cloud Run / Local
# -*- coding: utf-8 -*-

import os
import json
from io import StringIO
from typing import List, Dict, Any

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import streamlit as st
import pandas as pd
import math
import time

# =========================
# Config & HTTP client
# =========================

API_URL_DEFAULT = "http://127.0.0.1:8080"
API_URL = os.getenv("API_URL", None) or st.secrets.get("API_URL", API_URL_DEFAULT)

st.set_page_config(
    page_title="Analyse des sentiments éducatifs sur les évaluations des tuteurs",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def http_client(base_url: str):
    s = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.base_url = base_url.rstrip("/")
    s.timeout = 90
    return s

http = http_client(API_URL)

# =========================
# Helpers UI / data
# =========================

# F1 (en %) par modèle — adapter si besoin
F1_SCORES = {
    "apprentissage": 75.11,
    "valeur": 79.60,
    "rendement-scolaire": 79.80,
    "autogestion": 75.11,
    "sentiment-de-compétence": 71.00,
    "sentiment-de-autonomie": 75.00,
    "prise-de-decision-responsable": 67.00,
    "habillete-relationnelle": 72.00,
}

# Badge couleur selon F1
def f1_badge(f1: float) -> str:
    # > 78 : vert ; 74 <= F1 <= 78 : jaune ; < 74 : rouge
    if f1 > 78:
        return f"🟢 {f1:.2f}"
    if f1 >= 74:
        return f"🟡 {f1:.2f}"
    return f"🟠 {f1:.2f}"

# Concat colonnes pour CSV (onglets 3 et 4)
CSV_CONCAT_COLS = [
    "school","subject","attentive","group_discussion","follows_instructions",
    "completes_exercises","is_focused","open_to_new_activities","likes_challenges",
    "clearly_states_ideas","comments","to_work_on","language"
]

def _badge(status: str) -> str:
    if status == "ok":
        return "🟢 OK"
    if status == "initializing":
        return "🟡 Initialisation"
    return "🔴 KO"

def _clean_cell(x: Any) -> str:
    s = "" if x is None else str(x)
    return s.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()

def _concat_row_for_csv(row: pd.Series) -> str:
    parts: List[str] = []
    for c in CSV_CONCAT_COLS:
        if c in row.index:
            val = _clean_cell(row[c])
            if val:
                parts.append(val)
    txt = ". ".join(parts)
    if txt and not txt.endswith("."):
        txt += "."
    return txt

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

# =========================
# API calls
# =========================

def _call_health(api_url: str) -> Dict[str, Any]:
    r = http.get(f"{api_url.rstrip('/')}/health", timeout=20)
    r.raise_for_status()
    return r.json()

def _call_predict_label(api_url: str, model: str, text: str, threshold: float) -> Dict[str, Any]:
    payload = {"model": model, "text": text, "threshold": threshold}
    r = http.post(f"{api_url.rstrip('/')}/predict_label", json=payload, timeout=http.timeout)
    r.raise_for_status()
    return r.json()

def _call_predict_batch(api_url: str, model: str, texts: List[str], threshold: float) -> Dict[str, Any]:
    payload = {"model": model, "texts": texts, "threshold": threshold}
    r = http.post(f"{api_url.rstrip('/')}/predict_batch", json=payload, timeout=http.timeout)
    r.raise_for_status()
    return r.json()

def _call_predict_multi_single(api_url: str, models: List[str], text: str, threshold: float) -> Dict[str, Any]:
    payload = {"models": models, "text": text, "threshold": threshold}
    r = http.post(f"{api_url.rstrip('/')}/predict_multi", json=payload, timeout=http.timeout)
    r.raise_for_status()
    return r.json()

def _call_predict_multi_batch(api_url: str, models: List[str], texts: List[str], threshold: float) -> Dict[str, Any]:
    payload = {"models": models, "texts": texts, "threshold": threshold}
    r = http.post(f"{api_url.rstrip('/')}/predict_multi", json=payload, timeout=http.timeout)
    r.raise_for_status()
    return r.json()

# =========================
# Sidebar
# =========================

with st.sidebar:
    st.subheader("Source API")
    api_url = st.text_input("URL API", API_URL, help="Ex: https://api-tuteurs-xxxx-uc.a.run.app")
    st.caption("Tu peux éditer cette URL à la volée.")

    # Santé API + infos modèles
    try:
        meta = _call_health(api_url)
        status = meta.get("status", "unknown")
        models_meta = meta.get("models", [])
        st.success(f"API: {_badge(status)}")
        st.caption(f"Seuil par défaut: {meta.get('default_threshold', 0.5)}")

        if models_meta:
            st.markdown("**Modèles disponibles**")
            for m in models_meta:
                name = m.get("name")
                flag = "✅" if m.get("available") else ("⌛" if m.get("loading") else "❌")
                f1 = F1_SCORES.get(name)
                if f1 is not None:
                    st.write(f"- {flag} **{name}** · F1: {f1_badge(f1)} (pos_idx={m.get('pos_idx', '?')})")
                else:
                    st.write(f"- {flag} **{name}** (pos_idx={m.get('pos_idx', '?')})")
        else:
            st.warning("Aucun modèle détecté par /health.")
    except Exception as e:
        st.error(f"API: KO ({e})")
        st.stop()

    st.divider()
    st.markdown("**Choix du modèle (onglets 1 & 3)**")
    detected_models = [m.get("name") for m in models_meta if m.get("available")]
    default_model_idx = 0 if detected_models else None
    model_choice = st.selectbox(
        "Modèle (mono-modèle)",
        detected_models or ["apprentissage"],
        index=(default_model_idx if default_model_idx is not None else 0),
        help="Utilisé dans ‘Essai unitaire’ et ‘Batch CSV (mono-modèle)’"
    )

# =========================
# Header
# =========================

st.title("🧠 Analyse des sentiments éducatifs sur les évaluations des Enseignants")
st.caption("Frontend Streamlit • Backend Flask — Sélection de modèle, Formulaire FR, Batch CSV et Comparatif multi-modèles")

# =========================
# Onglets
# =========================

tab1, tab2, tab3, tab4 = st.tabs([
    "✨ Essai unitaire",
    "📝 Formulaire (FR)",
    "📂 Batch CSV (mono-modèle)",
    "⚖️ Comparatif multi-modèles (CSV)"
])

# -------------------------------------------------------------------
# Onglet 1 : Essai unitaire (Texte uniquement)
# -------------------------------------------------------------------
with tab1:
    st.subheader("Essai unitaire (texte)")
    colA, colB = st.columns([2, 1])
    with colA:
        txt = st.text_area(
            "Texte",
            height=160,
            placeholder="Ex: L'élève planifie seul ses tâches et respecte l'horaire."
        )
    with colB:
        thr = st.slider("Seuil (présence)", 0.0, 1.0, float(meta.get("default_threshold", 0.5)), 0.01)

    if st.button("Prédire", type="primary", use_container_width=True):
        if not txt.strip():
            st.warning("Merci de saisir un texte.")
        else:
            try:
                res = _call_predict_label(api_url, model_choice, txt.strip(), thr)
                st.metric(
                    "Décision",
                    "Présence" if res.get("label", 0) == 1 else "Absence",
                    delta=f"p={float(res.get('prob', 0.0)):.3f} • seuil={float(res.get('threshold', thr)):.2f}"
                )
                with st.expander("Réponse API"):
                    st.json(res)
            except Exception as e:
                st.error(f"Erreur: {e}")

# -------------------------------------------------------------------
# Onglet 2 : Formulaire FR only
# -------------------------------------------------------------------
with tab2:
    st.subheader("Formulaire d’observation (FR)")
    st.caption("Les réponses sont jointes en un seul texte (séparées par « . ») avant l’inférence. Uniquement des libellés francophones.")

    OPTS_FR = [
        "moins que les autres",
        "généralement",
        "souvent",
        "sans objet",
    ]

    def _clean(s):
        return ("" if s is None else str(s)).replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()

    def _form_to_text_fr(d):
        parts = []
        for _, v in d.items():
            sv = _clean(v)
            if sv:
                parts.append(sv)
        txt = ". ".join(parts)
        if txt and not txt.endswith("."):
            txt += "."
        return txt

    col1, col2 = st.columns(2)
    with col1:
        attentive              = st.selectbox("Attentif / attentive", OPTS_FR, index=1, key="f_attentive_fr")
        group_discussion       = st.selectbox("Discussion de groupe", OPTS_FR, index=1, key="f_group_fr")
        follows_instructions   = st.selectbox("Suit les consignes", OPTS_FR, index=1, key="f_follow_fr")
        completes_exercises    = st.selectbox("Termine les exercices", OPTS_FR, index=2, key="f_exos_fr")
    with col2:
        is_focused             = st.selectbox("Concentré·e", OPTS_FR, index=0, key="f_focus_fr")
        open_to_new_activities = st.selectbox("Ouvert·e à de nouvelles activités", OPTS_FR, index=1, key="f_open_fr")
        likes_challenges       = st.selectbox("Aime les défis", OPTS_FR, index=1, key="f_chal_fr")
        clearly_states_ideas   = st.selectbox("Exprime clairement ses idées", OPTS_FR, index=1, key="f_ideas_fr")

    comments = st.text_area(
        "Commentaires (facultatif)",
        height=160,
        key="f_comments_fr",
        placeholder="Observations libres, contexte, exemples concrets..."
    )

    st.divider()
    colA, colB, colC = st.columns(3)
    with colA:
        f_mode = st.radio("Mode", ["Modèle seul", "Comparatif (multi-modèles)"], horizontal=True, key="f_mode_fr")
    with colB:
        f_thr = st.slider("Seuil (présence)", 0.0, 1.0, float(meta.get("default_threshold", 0.5)), 0.01, key="f_thr_fr")
    with colC:
        detected = [m.get("name") for m in models_meta if m.get("available")]
        f_model = st.selectbox("Modèle (si ‘Modèle seul’)", detected or ["apprentissage"], index=0, key="f_model_fr")

    if st.button("Analyser le formulaire", type="primary", key="f_btn_fr"):
        form_fr = {
            "attentif_attentive": attentive,
            "discussion_de_groupe": group_discussion,
            "suit_les_consignes": follows_instructions,
            "termine_les_exercices": completes_exercises,
            "concentration": is_focused,
            "ouverture_aux_nouvelles_activites": open_to_new_activities,
            "aime_les_defis": likes_challenges,
            "expression_claire_des_idees": clearly_states_ideas,
            "commentaires": _clean(comments),
        }
        used_text = _form_to_text_fr(form_fr)

        try:
            if f_mode == "Modèle seul":
                res = _call_predict_label(api_url, f_model, used_text, f_thr)
                st.metric(
                    "Décision",
                    "Présence" if res.get("label", 0) == 1 else "Absence",
                    delta=f"p={float(res.get('prob', 0.0)):.3f} • seuil={float(res.get('threshold', f_thr)):.2f}"
                )
                with st.expander("Texte envoyé (FR)"):
                    st.code(used_text, language="text")
                with st.expander("Réponse API"):
                    st.json(res)
            else:
                # Comparatif sur le même texte
                model_names = detected or [m.get("name") for m in models_meta]
                res = _call_predict_multi_single(api_url, model_names, used_text, f_thr)
                st.success("Comparatif effectué.")
                st.write("Texte utilisé (FR) :")
                st.code(used_text, language="text")

                pm = res.get("per_model", {})
                cols = st.columns(min(3, len(model_names)) or 1)
                for i, name in enumerate(model_names):
                    block = pm.get(name, {})
                    with cols[i % len(cols)]:
                        if "prob" in block and "label" in block:
                            p = float(block["prob"])
                            y = int(block["label"])
                            st.metric(name, "Présence" if y == 1 else "Absence", delta=f"p={p:.3f}")
                        elif "error" in block:
                            st.error(f"{name} : {block.get('error')}")
                        else:
                            st.info(f"{name} : pas de données.")
        except Exception as e:
            st.error(f"Erreur : {e}")

# -------------------------------------------------------------------
# Onglet 3 : Batch CSV (mono-modèle)
# -------------------------------------------------------------------
with tab3:
    st.subheader("Batch CSV (mono-modèle)")
    st.caption("Le fichier doit contenir les colonnes à concaténer : " + ", ".join(CSV_CONCAT_COLS))
    thr_b = st.slider("Seuil (présence)", 0.0, 1.0, float(meta.get("default_threshold", 0.5)), 0.01, key="thr_b")
    file = st.file_uploader("Choisir un fichier CSV", type=["csv"], key="csv_batch")

    if file is not None:
        try:
            s = StringIO(file.getvalue().decode("utf-8"))
            df = pd.read_csv(s)
        except Exception as e:
            st.error(f"Impossible de lire le CSV: {e}")
            st.stop()

        # Vérif colonnes
        missing = [c for c in CSV_CONCAT_COLS if c not in df.columns]
        if missing:
            st.error(f"Colonnes manquantes: {missing}")
            st.stop()

        st.write("Aperçu (max 200 lignes) :")
        st.dataframe(df.head(200), use_container_width=True)

        if st.button("Lancer les prédictions (mono-modèle)", type="primary", use_container_width=True):
            texts = [ _concat_row_for_csv(row) for _, row in df.iterrows() ]
            texts = [t for t in texts if t.strip()]

            if not texts:
                st.warning("Aucun texte valide détecté.")
                st.stop()

            all_probs: List[float] = []
            CHUNK = 128
            total = len(texts)
            pbar = st.progress(0, text="Envoi des requêtes…")

            try:
                for idx, chunk in enumerate(chunked(texts, CHUNK)):
                    out = _call_predict_batch(api_url, model_choice, chunk, thr_b)
                    probs = out.get("per_model", {}).get(model_choice, {}).get("probs", [])
                    all_probs.extend(probs)
                    pct = min(1.0, ( (idx+1)*CHUNK ) / total)
                    pbar.progress(pct, text=f"Traitement… {min((idx+1)*CHUNK, total)}/{total}")
                    time.sleep(0.02)
            except Exception as e:
                st.error(f"Erreur pendant le batch: {e}")
                st.stop()
            finally:
                pbar.progress(1.0, text="Terminé")

            out_df = df.copy()
            out_df["text"] = texts[:len(out_df)]
            out_df["prob"] = (all_probs + [None]*len(out_df))[:len(out_df)]
            out_df["label"] = [1 if (p is not None and float(p) >= float(thr_b)) else (0 if p is not None else None) for p in out_df["prob"]]

            st.success("Prédictions terminées.")
            st.dataframe(out_df, use_container_width=True)

            st.download_button(
                "⬇️ Télécharger CSV (mono-modèle)",
                out_df.to_csv(index=False).encode("utf-8"),
                file_name="batch_monomodele.csv",
                mime="text/csv"
            )

# -------------------------------------------------------------------
# Onglet 4 : Comparatif multi-modèles (CSV)
# -------------------------------------------------------------------
with tab4:
    st.subheader("Comparatif multi-modèles (CSV)")
    st.caption("Même corpus, scoré par plusieurs modèles. Les colonnes sont concaténées comme dans l’onglet Batch.")
    detected_all = [m.get("name") for m in models_meta if m.get("available")]
    chosen_models = st.multiselect(
        "Modèles à comparer",
        options=detected_all or ["apprentissage"],
        default=detected_all or ["apprentissage"],
        help="Décoche les modèles que tu ne veux pas utiliser."
    )
    thr_c = st.slider("Seuil (présence)", 0.0, 1.0, float(meta.get("default_threshold", 0.5)), 0.01, key="thr_c")
    file_c = st.file_uploader("CSV pour comparatif", type=["csv"], key="csv_cmp")

    if file_c is not None:
        try:
            s = StringIO(file_c.getvalue().decode("utf-8"))
            dfc = pd.read_csv(s)
        except Exception as e:
            st.error(f"Impossible de lire le CSV: {e}")
            st.stop()

        missing = [c for c in CSV_CONCAT_COLS if c not in dfc.columns]
        if missing:
            st.error(f"Colonnes manquantes: {missing}")
            st.stop()

        st.write("Aperçu (max 200 lignes) :")
        st.dataframe(dfc.head(200), use_container_width=True)

        if st.button("Comparer (multi-modèles)", type="primary", use_container_width=True):
            texts_c = [ _concat_row_for_csv(row) for _, row in dfc.iterrows() ]
            texts_c = [t for t in texts_c if t.strip()]
            if not texts_c:
                st.warning("Aucun texte valide dans le CSV.")
                st.stop()

            if not chosen_models:
                st.warning("Aucun modèle sélectionné.")
                st.stop()

            out_compare = pd.DataFrame({"text_used": texts_c})
            CHUNK = 96
            total = len(texts_c)
            steps = math.ceil(total / CHUNK)
            pbar = st.progress(0, text="Envoi des requêtes comparatives…")

            per_model_probs: Dict[str, List[float]] = {m: [] for m in chosen_models}
            per_model_labels: Dict[str, List[int]] = {m: [] for m in chosen_models}

            try:
                for i, chunk in enumerate(chunked(texts_c, CHUNK)):
                    res = _call_predict_multi_batch(api_url, chosen_models, chunk, thr_c)
                    pm = res.get("per_model", {})
                    for m in chosen_models:
                        block = pm.get(m, {})
                        probs = block.get("probs", []) or []
                        preds = block.get("preds", []) or []
                        per_model_probs[m].extend(probs)
                        per_model_labels[m].extend(preds)
                    pbar.progress((i+1)/steps, text=f"Traitement… bloc {i+1}/{steps}")
                    time.sleep(0.02)
            except Exception as e:
                st.error(f"Erreur en comparatif: {e}")
                st.stop()
            finally:
                pbar.progress(1.0, text="Terminé")

            # Compose le tableau final
            for m in chosen_models:
                probs = per_model_probs[m]
                preds = per_model_labels[m]
                out_compare[f"prob({m})"] = (probs + [None]*len(out_compare))[:len(out_compare)]
                out_compare[f"label({m})"] = (preds + [None]*len(out_compare))[:len(out_compare)]

            st.success("Comparatif terminé.")
            st.dataframe(out_compare, use_container_width=True)
            st.download_button(
                "⬇️ Télécharger CSV comparatif",
                out_compare.to_csv(index=False).encode("utf-8"),
                file_name="comparatif_multi_modeles.csv",
                mime="text/csv"
            )

# =========================
# Footer
# =========================

st.divider()
st.caption("© UI Streamlit — Cloud Run • API Flask — F1 & badges intégrés, formulaires FR, batch & comparatif.")
