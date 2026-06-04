# Stage 1: build frontend
FROM node:22-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: final image
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-server \
    nginx \
    supervisor \
    git \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app/backend

# Install Python deps first (layer cache)
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/backend/.venv/bin:$PATH"

# Copy application code and models
COPY backend/app/ ./app/
COPY backend/models/ ./models/

# Frontend static files
COPY --from=frontend-build /frontend/dist /usr/share/nginx/html

# Config files
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/app.conf

RUN mkdir -p /var/log/supervisor /app/backend/uploads /app/backend/data \
    && ln -sf /dev/stdout /var/log/nginx/access.log \
    && ln -sf /dev/stderr /var/log/nginx/error.log

EXPOSE 80

CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
