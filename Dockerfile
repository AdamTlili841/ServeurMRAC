FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    CHECKPOINT_PATH=/app/best_model.pt \
    PRELOAD_MODEL=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY multimodal_model.py main.py start.sh ./
COPY best_model.pt ./

# Télécharge ViT + BERT au build (évite timeout / crash au premier /predict).
RUN python -c "\
from transformers import BertModel, BertTokenizer, ViTImageProcessor, ViTModel; \
BertTokenizer.from_pretrained('bert-base-uncased'); \
BertModel.from_pretrained('bert-base-uncased'); \
ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k'); \
ViTModel.from_pretrained('google/vit-base-patch16-224-in21k'); \
print('Hugging Face weights cached.')"

RUN chmod +x /app/start.sh

EXPOSE 10000

CMD ["/app/start.sh"]
