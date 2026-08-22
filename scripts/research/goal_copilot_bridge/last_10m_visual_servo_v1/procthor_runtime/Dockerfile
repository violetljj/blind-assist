FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libnss3 libvulkan1 mesa-vulkan-drivers vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir ai2thor==5.0.0 numpy==1.26.4 pillow==11.3.0

WORKDIR /work
COPY smoke_procthor.py /work/smoke_procthor.py
ENTRYPOINT ["python", "/work/smoke_procthor.py"]
