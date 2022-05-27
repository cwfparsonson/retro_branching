FROM nvcr.io/nvidia/tensorflow:21.06-tf2-py3

RUN apt-get update && apt-get install git libcusolver10 -y

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      curl \
      git \
      wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# install in batch (silent) mode, does not edit PATH or .bashrc or .bash_profile
# -p path
# -f force
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH=/root/miniconda3/bin:${PATH}

# Install conda requirements into conda base.
COPY ./retro_branching.yaml /tmp/retro_branching.yaml
RUN conda env update -n base --file /tmp/retro_branching.yaml && \
  rm -rf /tmp/*
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# Install pip packages into conda base.
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --quiet -r /tmp/requirements.txt && rm -rf /tmp/*

# Upgrade numpy for compatability
RUN pip install numpy --upgrade

# Configure paths.
RUN retro_branching 'export PATH=/opt/conda/bin:$PATH' >> .bashrc
RUN retro_branching 'export PATH=$HOME/bin:$HOME/.local/bin:$PATH' >> .bashrc
RUN retro_branching 'source /opt/conda/etc/profile.d/conda.sh' >> .bashrc
#RUN torch "alias ls='ls --color=auto'" >> .bashrc

# Tensorflow
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV TF_CPP_MIN_LOG_LEVEL=3

# working directory
WORKDIR /home/app/retro_branching
ENV PYTHONPATH=$PWD:$PYTHONPATH






