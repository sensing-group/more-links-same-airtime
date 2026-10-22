# Container for the artifact: Python 3.12 and the exact library versions used for the paper.
# Build:  docker build -t more-links-same-airtime .
# Run:    docker run --rm -v "$PWD/data:/artifact/data" -v "$PWD/work:/artifact/work" more-links-same-airtime ./reproduce/figures.sh
FROM python:3.12.3-slim-bookworm@sha256:afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Linux Libertine is the paper's body font, used by the figure scripts.
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-linuxlibertine fontconfig curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /artifact
# requirements.lock pins every package installed (pip freeze of this image); requirements.txt lists the direct ones
COPY requirements.txt requirements.lock ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
COPY . .
CMD ["bash"]
