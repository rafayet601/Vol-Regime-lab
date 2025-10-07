# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/interim data/processed artifacts reports/figures logs

# Set up pre-commit hooks (optional)
RUN pre-commit install || true

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Default command
CMD ["python", "-c", "import regime_lab; print('Regime Lab is ready!')"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import regime_lab; print('Health check passed')" || exit 1

# Expose port (if needed for future API development)
EXPOSE 8000

# Labels
LABEL maintainer="Your Name <your.email@example.com>"
LABEL description="S&P 500 regime detection using Student-t HMM"
LABEL version="0.1.0"
