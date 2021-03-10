FROM tensorflow/tensorflow:1.15.0rc2-gpu-jupyter

WORKDIR /tf/

RUN apt-get update && apt-get install -y git wget

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip -q awscliv2.zip && \
    ./aws/install

ENV HOME=/tf/
RUN python -c "from tensorflow.keras.applications.vgg16 import VGG16; VGG16()"
RUN chmod -R go+w /tf/.keras

RUN pip install \
    requests \
    pillow \
    git+https://github.com/uwmisl/cupyck \
    tqdm \
    h5py \
    tables \
    unireedsolomon

RUN pip install seaborn
