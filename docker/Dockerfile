FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]

# System packages 
RUN apt update --fix-missing
RUN apt install -y software-properties-common wget curl gpg gcc git make libssl-dev libmodule-install-perl libboost-all-dev

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN conda install pip
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

RUN python -c "import torch; print(torch.__version__)"
RUN nvcc --version

# Install new version of CMake needed for spconv.
RUN apt purge --auto-remove cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN lsb_release -a
RUN apt update
RUN apt install -y cmake

# Install prerequisites needed for spconv and second.pytorch.
RUN apt install -y libboost-all-dev libgl1-mesa-dev
RUN conda install scikit-image scipy numba pillow matplotlib seaborn psutil
RUN pip install fire tensorboardX protobuf opencv-python nuscenes-devkit

# Install spconv
RUN git clone --recurse-submodules https://github.com/kylevedder/spconv.git spconv
WORKDIR spconv
ENV SPCONV_FORCE_BUILD_CUDA="1"
RUN python setup.py bdist_wheel
RUN pip install dist/spconv-1.2.1-cp38-cp38-linux_x86_64.whl
WORKDIR /
RUN python -c "import spconv"

ENV PYTHONPATH="/second.pytorch::$PYTHONPATH"
WORKDIR /second.pytorch/second
