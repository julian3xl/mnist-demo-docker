FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
LABEL maintainer="julian3xl <julian3xl@gmail.com>"

RUN apt-get update && apt-get install -y libcudnn8 python3-pip vim wget

# CUDA 11.2 patch
RUN ln -s /usr/local/cuda-11.2/lib64/libcusolver.so.11 /usr/local/cuda-11.2/lib64/libcusolver.so.10 && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64/

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda-11.2/lib64/

COPY requirements.txt /tmp/

RUN pip install -U -r /tmp/requirements.txt

COPY src/* /src/

WORKDIR /src

ENTRYPOINT [ "python3" ]