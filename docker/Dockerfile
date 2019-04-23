FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
    git \
    python3.6 \
    python3-pip \
    python3-dev \
    mecab \
    mecab-ipadic-utf8

WORKDIR /opt
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
  && cd mecab-ipadic-neologd \
  && ./bin/install-mecab-ipadic-neologd -n -y \
  && cd .. \
  && rm -rf mecab-ipadic-neologd

RUN pip3 install --upgrade pip

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

ARG UID
RUN useradd docker -l -u $UID -s /bin/bash -m
USER docker

WORKDIR /work

ENTRYPOINT ["/bin/bash"]
