FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install minimal OS utilities needed for compiling C-extensions or tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with UID 1000 for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user

WORKDIR /app

# Upgrade build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install production dependencies with layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code with user ownership
COPY --chown=user:user . .

# Switch to non-root user
USER user

# Configure environment path for user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]

