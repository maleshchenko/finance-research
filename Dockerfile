FROM python:3.11-slim

# System dependencies for pdfplumber / camelot
RUN apt-get update && apt-get install -y --no-install-recommends \
        ghostscript \
        libpoppler-dev \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --break-system-packages ".[dev]"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
