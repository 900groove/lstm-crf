FROM ubuntu:16.04

RUN apt update && apt install -y \
    git \
    make \
    curl \
    wget \
    xz-utils \
    sudo \
    file \
    python3.5 \
    python3-pip \
    python3-dev \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8

RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git\
    && cd mecab-ipadic-neologd\
    && bin/install-mecab-ipadic-neologd -n -y

RUN pip3 install --upgrade pip

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ARG UID
RUN useradd docker -l -u $UID -s /bin/bash -m
USER docker

WORKDIR /work

ENTRYPOINT ["/bin/bash"]
