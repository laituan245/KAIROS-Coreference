# FROM ubuntu:latest
FROM nvidia/cuda:10.2-base-ubuntu16.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 LANGUAGE=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

ARG DEBIAN_FRONTEND=noninteractive

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    python3-pip \
    unzip \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN /opt/conda/bin/conda create -n aida_coreference python=3.6 && \
    /opt/conda/envs/aida_coreference/bin/pip install torch torchvision && \
    /opt/conda/envs/aida_coreference/bin/pip install -r requirements.txt && \
    /opt/conda/envs/aida_coreference/bin/python3.6 -m nltk.downloader punkt && \
    /opt/conda/envs/aida_coreference/bin/python3.6 -m nltk.downloader treebank && \
    /opt/conda/envs/aida_coreference/bin/python3.6 -m nltk.downloader conll2000 && \
    /opt/conda/envs/aida_coreference/bin/python3.6 -m nltk.downloader wordnet && \
    /opt/conda/envs/aida_coreference/bin/python3.6 -m nltk.downloader averaged_perceptron_tagger

RUN /opt/conda/bin/conda clean -tipsy
RUN chmod +x ./wait-for-it.sh

CMD ["/bin/bash"]
