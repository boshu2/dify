# Sandbox Service Dockerfile
# Isolated environment for safe code execution

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for JavaScript execution
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -s /bin/bash sandbox

# Set working directory
WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system -e .

# Copy application code
COPY app/ ./app/

# Create temp directory with proper permissions
RUN mkdir -p /tmp/sandbox && chown -R sandbox:sandbox /tmp/sandbox

# Switch to non-root user
USER sandbox

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SANDBOX_HOST=0.0.0.0
ENV SANDBOX_PORT=8194

# Expose port
EXPOSE 8194

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8194/health || exit 1

# Run the service
CMD ["python", "-m", "app.main"]
