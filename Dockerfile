FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /agent

# Install dependencies first so this layer is cached between code changes
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code (see .dockerignore for what is excluded)
COPY src ./src

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /agent
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health')" || exit 1

# Shell form so $PORT is expanded (works on Render, Railway, Cloud Run, etc.)
CMD uvicorn src.agent.main:app --host 0.0.0.0 --port ${PORT}