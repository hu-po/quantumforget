FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN pip install --upgrade pip
RUN pip install \
    torchvision \
    timm \
    Pillow \
    pyzmq
WORKDIR /workspace/quantumforget
RUN python setup.py install
COPY ./*.py /workspace/quantumforget