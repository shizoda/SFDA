# https://hub.docker.com/r/nvidia/cuda/tags 
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
RUN apt-get update
RUN apt-get install -y sudo git python3-pip vim htop libgl1-mesa-dev libsm6 libxrender1 zsh pigz
RUN chsh -s $(which zsh)
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install nibabel scipy pandas tqdm termcolor protobuf matplotlib scikit-image
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install torch==1.8.1+cu111 torchvision -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install nibabel==4.0.1 numpy==1.20.0 pandas==1.1.0
RUN pip install --upgrade scikit-learn SimpleITK
RUN pip install kornia
RUN pip install cupy-cuda110
RUN apt-get -y update
RUN pip install torchviz opencv-python
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y graphviz
RUN pip install h5py
RUN pip install --upgrade torchvision
RUN pip install nnunet
RUN pip install onnx einops monai
RUN apt-get install -y hdf5-tools
ENV CUDA_DEVICE_ORDER="PCI_BUS_ID"

