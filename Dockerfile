FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

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
    glibc-source 


RUN apt-get autoremove -y && \
    apt-get autoclean -y && \
    apt-get clean -y  && \
    rm -rf /var/lib/apt/lists/*

ENV WRKSPCE="/workspace"

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p $WRKSPCE/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="$WRKSPCE/miniconda3/bin:${PATH}"


COPY requirements.yml .
RUN conda update -n base -c defaults conda
RUN conda install python=3.9
RUN conda env update -n base --file requirements.yml

RUN pip install --upgrade ultralytics
RUN pip install wandb
RUN conda clean -y --all
