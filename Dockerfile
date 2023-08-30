FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN pip install --upgrade pip
RUN pip install \
    torchvision \
    timm \
    Pillow \
    matplotlib \
    numpy \
    pennylane \
    tensorflow \
    torch \
    pyyaml \
    hyperopt \
    tensorboardX
WORKDIR /workspace/quantumforget
COPY ./*.py /workspace/quantumforget/