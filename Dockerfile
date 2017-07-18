FROM ubuntu:14.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    libfreetype6-dev \
    libxft-dev  \
    libpng12-dev \
    libxml2-dev \
    libxslt1-dev  \
    libjpeg8 \
    libjpeg62-dev \
    libfreetype6  \
    libfreetype6-dev \
    libwebp-dev

RUN apt-get install -y --no-install-recommends \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libdc1394-22-dev \
    && \
    apt-get -y build-dep python-imaging && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build && \
    /opt/conda/bin/conda create -y --name py3 python=3.4.3 numpy pyyaml scipy cython scikit-image \
                                              scikit-learn ipython matplotlib mkl && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/envs/py3/bin:$PATH
RUN conda install --name py3 --channel https://conda.anaconda.org/menpo opencv3
