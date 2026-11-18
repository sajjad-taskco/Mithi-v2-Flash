# Mithi v2 Flash — CPU text-to-speech (11 Indian languages, 10 voices, 24 kHz)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (much smaller than the CUDA wheel)
RUN pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cpu

RUN pip install "transformers>=5.0" "safetensors>=0.4" "sentencepiece>=0.2" "numpy>=1.26" "soundfile>=0.12" \
        "misaki[en]>=0.9.4" "espeakng-loader>=0.2.4" "phonemizer-fork>=3.3.2" "spacy>=3.8" "num2words>=0.5" \
        "huggingface_hub>=0.34" "fastapi>=0.115" "uvicorn[standard]>=0.30" \
    && python -m spacy download en_core_web_sm

# Model weights, pinned to a commit (skip the demo video)
ARG MODEL_REVISION=1fc0dd68496a2483508ec764d310146c41ff90be
RUN hf download Oomni/mithi-v2-flash --revision ${MODEL_REVISION} --exclude "assets/*" --local-dir /app \
    && rm -rf /app/.cache ${HF_HOME}

WORKDIR /app
COPY server.py /app/server.py

RUN useradd -m -u 1000 mithi && mkdir -p /out /data && chown -R mithi /app /out /data
USER mithi

# Everything is local from here on
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MITHI_THREADS=4 MITHI_DATA=/data
# The API key lives here (generated on first start) — keep it on a volume
VOLUME /data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
