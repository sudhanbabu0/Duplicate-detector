FROM python:3.10-slim

# Create non-root user with UID 1000 (standard for Hugging Face Spaces & secure containers)
RUN useradd -m -u 1000 user

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache SentenceTransformer model inside the image so it starts instantly
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application files and grant full permissions to user
COPY --chown=user:user . /app
RUN chmod -R 777 /app

# Switch to non-root user
USER user
ENV PORT=7860 \
    PYTHONUNBUFFERED=1

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Run Uvicorn (binds to $PORT or default 7860 for Hugging Face Spaces, 8000/10000 for Render)
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
