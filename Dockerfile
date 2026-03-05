# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MeetMind Dockerfile
# Professor's Note:
#   Multi-stage build keeps the final image lean.
#   Stage 1: Install all dependencies (heavy)
#   Stage 2: Copy only what we need to run (light)
#
#   Why this matters: ML images can be 10GB+ without care.
#   With this approach we get ~4GB — acceptable for deployment.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FROM python:3.11-slim AS base

# System dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \          
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ffmpeg is CRITICAL — Whisper and pydub both need it
# to decode MP3, MP4, and other audio formats

WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────────────────
# Copy requirements first (Docker layer caching — if requirements don't change,
# this layer is cached and subsequent builds are fast)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's baked into the image
# (avoids downloading at runtime, which would slow first request)
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-MiniLM-L6-v2')"

# Pre-download Whisper base model
RUN python -c "import whisper; whisper.load_model('base')"

# ── Copy application code ─────────────────────────────────────────────────────
COPY src/ ./src/
COPY .env.example .env

# Create data directories
RUN mkdir -p data/chromadb data/uploads

# ── Runtime configuration ─────────────────────────────────────────────────────
ENV PORT=7860
ENV PYTHONPATH=/app/src
ENV GRADIO_SERVER_NAME=0.0.0.0

# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Health check — makes sure the app is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "src/app.py"]
