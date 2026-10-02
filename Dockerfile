# Two stages: Node builds the interface to static files, Python serves both the
# API and those files as one Cloud Run service. One service means no CORS and
# one deployment to keep in step.

FROM node:22-slim AS ui
WORKDIR /ui
COPY app/frontend/package.json app/frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY app/frontend/ ./
RUN npm run build

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

COPY app/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# src/ holds the four pipelines and stays the single source of truth; the
# backend imports from it rather than keeping a second copy that can drift.
COPY src/ ./src/
COPY app/backend/ ./app/backend/
COPY --from=ui /ui/out ./app/backend/static

WORKDIR /srv/app/backend

EXPOSE 8080
# Cloud Run supplies $PORT. One worker: the models load once and the work is
# CPU-bound, so concurrency comes from Cloud Run starting more instances.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
