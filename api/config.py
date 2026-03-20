# -*- coding: utf-8 -*-
import os
from pathlib import Path

# Seuil par défaut
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))

# Port (Cloud Run injecte PORT)
PORT = int(os.getenv("PORT", "8080"))

# Chargement des modèles au boot
USE_MODELS = os.getenv("USE_MODELS", "1") not in ("0", "false", "False")

# Racine des modèles
MODEL_ROOT = Path(os.getenv("MODEL_ROOT", str(Path(__file__).parent / "models")))

# Contrôles de threads (CPU Cloud)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Longueur max tokenizer
DEFAULT_MAX_LENGTH = 256

# Colonnes pour concat CSV (non fusionné)
Q_COLUMNS = [f"question_{i}" for i in range(24, 37)]
HOURS_COL = "hours_completed_on_creation"

# Liste officielle des 8 modèles
MODEL_NAMES = [
    "apprentissage",
    "autogestion",
    "habillete-relationnelle",
    "prise-de-decision-responsable",
    "rendement-scolaire",
    "sentiment-de-autonomie",
    "sentiment-de-compétence",
    "valeur",
]
