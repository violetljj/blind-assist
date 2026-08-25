FROM python:3.11-slim-bookworm@sha256:fe556eaffd0d96abd8685b83bb8a69addd0dc730fa193669a7df0d49ddcfac02

RUN apt-get update && apt-get install -y --no-install-recommends \
    libasound2 libglib2.0-0 libgl1 libgl1-mesa-dri libglu1-mesa libgtk-3-0 \
    libnss3 libxcomposite1 libxcursor1 libxi6 libxrandr2 libxss1 libxtst6 \
    mesa-utils xauth xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir ai2thor==5.0.0 numpy==1.26.4 pillow==11.3.0

ENV PYTHONUNBUFFERED=1
WORKDIR /work
COPY run_xvfb.sh /usr/local/bin/run-grail-xvfb
RUN chmod 0755 /usr/local/bin/run-grail-xvfb
ENTRYPOINT ["/usr/local/bin/run-grail-xvfb"]
CMD ["python", "/work/run_grail_procthor_native_m0.py"]
