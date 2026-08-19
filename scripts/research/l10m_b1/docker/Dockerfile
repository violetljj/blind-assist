# Pinned Codex searcher runtime.  The evaluator and hidden cohort never enter
# this image; they remain owned by the local controller.
FROM node:24.16.0-bookworm-slim

ARG CODEX_VERSION=0.148.0

ENV CODEX_HOME=/root/.codex \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SSL_CERT_FILE=/etc/ssl/certs/we1.crt.pem

COPY we1.crt.pem /etc/ssl/certs/we1.crt.pem

RUN mkdir -p /root/.codex /workspace /tmp/codex \
    && npm install --global --omit=dev "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force \
    && codex --version

WORKDIR /workspace
CMD ["codex"]
