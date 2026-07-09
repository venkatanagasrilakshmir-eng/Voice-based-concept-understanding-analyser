# VBCUA - Voice-Based Concept Understanding Analyser
# Dockerfile for the Streamlit frontend (default entrypoint).
# To containerize the FastAPI service instead, override CMD (see docker-compose.yml).

FROM python:3.11-slim

# System dependencies required by librosa/soundfile (libsndfile) and audio codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads reports data

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
