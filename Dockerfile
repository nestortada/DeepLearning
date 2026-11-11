FROM node:24-bookworm-slim AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.14-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    GLAUCOAI_CPU=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-backend.txt ./
RUN python -m pip install --no-cache-dir -r requirements-backend.txt \
    && python -m pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 \
        --index-url https://download.pytorch.org/whl/cpu
COPY backend/ ./backend/
COPY src/ ./src/
COPY configs/glaucoai-models.json configs/papila-internal-test.json ./configs/
COPY --from=frontend-build /build/dist ./frontend/dist/
RUN test -s backend/predict/resnet34.pt && test -s backend/predict/best.pt
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --start-period=60s CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/api/health', timeout=3).close()"]
CMD ["python", "-m", "backend.container"]
