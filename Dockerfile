FROM nvidia/cuda:12.6.2-base-ubuntu22.04

RUN apt-get -y update && apt-get install -y apt-transport-https

RUN apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    rsync \
    ffmpeg \
    htop \
    nano \
    libatlas-base-dev \
    libboost-all-dev \
    libeigen3-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopenblas-dev \
    libopenblas-base \
    libsm6 \
    libxext6 \
    libxrender-dev \
    glibc-source \
    python3 \
    python3-venv \
    python3-pip

RUN apt-get autoremove -y && \
    apt-get autoclean -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

ENV VENV_PATH="/opt/venv"
RUN python3 -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt