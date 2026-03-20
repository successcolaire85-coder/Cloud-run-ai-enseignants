# -*- coding: utf-8 -*-
"""
Point d’entrée Flask pour l’API (local & Cloud Run).
"""
from flask import Flask
import signal, sys, time
from contextlib import suppress

from config import USE_MODELS, PORT
from models import load_all_models_blocking, log_models_overview
from routes import api_bp
from utils import log

def create_app() -> Flask:
    app = Flask(__name__)
    # charge tous les modèles au boot si activé
    if USE_MODELS:
        load_all_models_blocking()
        log_models_overview()
    # enregistre les routes
    app.register_blueprint(api_bp)
    return app

app = create_app()

def _graceful_shutdown(*_):
    log("[SHUTDOWN] Signal reçu, arrêt gracieux…")
    with suppress(Exception):
        time.sleep(0.2)
    sys.exit(0)

signal.signal(signal.SIGTERM, _graceful_shutdown)

if __name__ == "__main__":
    log(f"[BOOT] USE_MODELS={USE_MODELS}  PORT={PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True, use_reloader=False)
