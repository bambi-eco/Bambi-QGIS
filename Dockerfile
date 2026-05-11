FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    bandit \
    detect-secrets \
    flake8

# Code is mounted at runtime via docker-compose volume — no COPY needed.
CMD ["bash", "run_checks.sh"]
