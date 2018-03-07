FROM nvidia/cuda:9.0-cudnn7-devel
LABEL maintainer="Álvaro Barbero Jiménez"
ARG INSTALL=install

# Install system dependencies
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  	build-essential \
  	curl \
  && apt-get clean

# Install python miniconda3 + requirements
ENV MINICONDA_HOME="/opt/miniconda"
ENV PATH="${MINICONDA_HOME}/bin:${PATH}"
RUN curl -o Miniconda3-latest-Linux-x86_64.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && chmod +x Miniconda3-latest-Linux-x86_64.sh \
  && ./Miniconda3-latest-Linux-x86_64.sh -b -p "${MINICONDA_HOME}" \
  && rm Miniconda3-latest-Linux-x86_64.sh
WORKDIR /root
COPY pip.txt pip.txt
COPY conda.txt conda.txt
COPY conda-gpu.txt conda-gpu.txt
COPY Makefile Makefile
RUN make ${INSTALL}
RUN conda clean -y -i -l -p -t

# App files
RUN mkdir neurowriter
COPY *.py /neurowriter/
COPY neurowriter /neurowriter/neurowriter
COPY corpus /neurowriter/corpus
WORKDIR /neurowriter

# Define locale
ENV LANG C.UTF-8  
ENV LC_ALL C.UTF-8
