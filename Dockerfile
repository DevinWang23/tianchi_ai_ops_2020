# Python3.7.4, cuda10.1 and pytorch1.4

FROM registry.cn-shanghai.aliyuncs.com/mengqiu/tianchi_competition:pytorch1.4-cuda10.1-py3

ENV PYTHONIOENCODING=UTF-8
ENV LANG C.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL C.UTF-8

ADD . /

WORKDIR /

CMD ["sh", "run.sh"]

