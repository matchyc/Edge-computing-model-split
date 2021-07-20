FROM docker.io/pytorch/pytorch:latest
WORKDIR /code
COPY ./src ./src
COPY ./modelparam ./modelparam
# RUN pip install -r requirements.txt
# RUN pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip config set global.index-url 'https://mirrors.aliyun.com/pypi/simple'
RUN sed -i "s@http://deb.debian.org@http://mirrors.aliyun.com@g" /etc/apt/sources.list
RUN python -m pip install --upgrade pip
# RUN pip install torch==1.5.1 torchvision==0.6.1
RUN python -m pip install grpcio
RUN python -m pip install grpcio-tools
CMD ["python", "src/server.py"]