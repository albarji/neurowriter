#FROM nvidia/cuda:8.0-devel
FROM nvidia/cuda

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
COPY conda.txt /root/conda.txt
COPY pip.txt /root/pip.txt
ENV ACCEPT_INTEL_PYTHON_EULA=yes
RUN conda install -q -y --file=/root/conda.txt \
  && conda clean -y -i -l -p -t \
  && pip install -r /root/pip.txt

# Install requirements
#ADD conda.txt pip.txt ./
#RUN conda install -y --file conda.txt && rm conda.txt
#RUN pip install -r pip.txt && rm pip.txt

# Clone project files
RUN mkdir neurowriter
COPY . /neurowriter
WORKDIR /neurowriter

# Launche Jupyter notebook with appropriate options
CMD jupyter notebook --allow-root --no-browser --ip='*'

