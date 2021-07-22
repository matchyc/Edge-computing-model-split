FROM docker.io/pytorch/pytorch:latest
WORKDIR /code
COPY ./src ./src
COPY ./modelparam ./modelparam
RUN pip config set global.index-url 'https://mirrors.aliyun.com/pypi/simple'
RUN sed -i "s@http://deb.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list
RUN python -m pip install --upgrade pip
RUN python -m pip install grpcio
RUN python -m pip install grpcio-tools
CMD ["python", "src/server.py"]