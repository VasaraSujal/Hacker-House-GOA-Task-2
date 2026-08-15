FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models/huggingface \
    TRANSFORMERS_CACHE=/opt/models/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.prod.txt /app/backend/requirements.prod.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install -r /app/backend/requirements.prod.txt

# Cache the serving embedding model in the immutable image. No API secrets are
# required or copied at build time.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY backend/app /app/backend/app
COPY backend/rag /app/backend/rag
COPY backend/evaluation /app/backend/evaluation
COPY data/indexes/bm25.pkl /app/data/indexes/bm25.pkl

WORKDIR /app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl --fail http://127.0.0.1:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
