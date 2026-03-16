FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN adduser --disabled-password --gecos "" --uid 1001 appuser

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --upgrade pip && pip install .
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 18080

CMD ["python", "-m", "proxy_api_server.server"]
