FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY multimodal_model.py main.py start.sh ./
COPY best_model.pt ./

RUN chmod +x /app/start.sh

ENV CHECKPOINT_PATH=/app/best_model.pt

# Render injecte PORT (souvent 10000) — l'app doit écouter sur cette variable.
EXPOSE 10000

CMD ["/app/start.sh"]
