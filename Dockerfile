# Python3.7, cuda10.0 and cuDNN7 (GPU)

FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04

MAINTAINER MengQiu Wang "wangmengqiu@ainnovation.com"

ENV REFRESHED_AT 2019-05-22
ENV PYTHONIOENCODING=UTF-8
ENV LANG C.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL C.UTF-8

USER root

# install packages
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends\
    software-properties-common \
    gcc \
	g++\
	wget\
	make \
	zlib1g \
	zlib1g-dev \
	build-essential \
	openssl \
	curl \
	libssl-dev \
	libffi-dev \
	vim \
	libbz2-dev \
	python3-tk \
	tk-dev \
	liblzma-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# install python 3.7.5
RUN wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
RUN tar -xvf Python-3.7.5.tgz
RUN cd Python-3.7.5 \
	&& ./configure \
	&& make \
	&& make install

# install pip3
RUN curl https://bootstrap.pypa.io/get-pip.py | python3

# install python packages listed in basic_requirements.txt
RUN mkdir /tools
COPY ./requirements.txt /tools
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev
RUN pip install --no-cache-dir -r /tools/requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# create a link to use python3 as as 'python'
RUN ln -s /usr/local/bin/python3 /usr/bin/python

# set up work directory
WORKDIR time_series_pipeline
COPY . .
RUN mkdir ./data
RUN mkdir ./data/output
RUN mkdir ./data/output/log
RUN mkdir ./figs

