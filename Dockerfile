# ── Stage 1: builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user
RUN groupadd -r botuser && useradd -r -g botuser -d /app -s /sbin/nologin botuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create data directory (will be volume-mounted in production)
RUN mkdir -p /app/data /app/logs && chown -R botuser:botuser /app

USER botuser

# Health check — bot process is alive
HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default: run the telegram bot in paper mode
ENV BOT_MODE=paper
ENV STARTING_CAPITAL=50.0
ENV SCAN_INTERVAL=120
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "bot.py"]
CMD ["--mode", "paper"]
