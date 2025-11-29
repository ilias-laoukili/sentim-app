# Dockerfile for Sentim App

FROM python:3.11-slim

# Install system dependencies (including curl for healthcheck)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
# Models are downloaded at runtime from Hugging Face Hub; do not copy local models

# --- CRITICAL CHANGES FOR HUGGING FACE ---
EXPOSE 7860
# Healthcheck must be a single line; add sensible defaults
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD curl --fail http://localhost:7860/_stcore/health || exit 1
CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
