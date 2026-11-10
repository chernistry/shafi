# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Runtime libs:
# - tini/ca-certificates: process + TLS
# - libxcb/libgl/glib/x* libs: required by cv2 pulled by docling_ibm_models (TableFormer)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
        libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --home-dir /home/appuser --shell /usr/sbin/nologin appuser

COPY pyproject.toml README.md ./
COPY src ./src

# Editable install keeps prompt markdown files available from source tree.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --upgrade pip \
    && pip install -e . \
    && mkdir -p /usr/local/lib/python3.12/site-packages/rapidocr/models \
    && chown -R appuser:appuser /usr/local/lib/python3.12/site-packages/rapidocr/models

USER appuser

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "shafi"]
