FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app
COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY . /app

# Démarrage Cloud Run : on sert l’app Flask définie dans app.py
# Optionnel: ajoute --preload si tu veux charger les modèles AVANT d’accepter du trafic
CMD gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT app:app --timeout 180
# CMD gunicorn --preload -w 2 -k gthread -b 0.0.0.0:$PORT app:app --timeout 180
