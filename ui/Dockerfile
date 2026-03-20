
# UI Dockerfile (Streamlit)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt

# Copier le code
COPY app.py /app/app.py
COPY settings.py /app/settings.py
COPY client.py /app/client.py
COPY utils.py /app/utils.py
COPY views.py /app/views.py

# Variables (surchargées au déploiement)
ENV API_URL=""
ENV PORT=8080

EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8080", "--browser.gatherUsageStats=false"]
