# mainly for MNIST-alike model
# demo use MNIST and simple Convlutional neural network
# Date: 2021 - 07 - 20
# Meng_chen@bupt.edu.cn
# gRPC server

import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import torch.tensor as tensor
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import model_split_pb2
import model_split_pb2_grpc
import grpc
from concurrent import futures
import logging
import pickle
import uuid
from model_configuration import *
import codecs
import io

class InferServicer(model_split_pb2_grpc.InferServicer):
    # implement
    def __init__(self) -> None:
        super().__init__()

    def infer_part(self, request, context):
        try:
            model = model_dict[request.model_id]
            model_param_path = '../modelparam/' + request.model_id.upper() +'.pt'
            manager.load_params(model, model_param_path)
            buffer = io.BytesIO(request.tensor_content)
            # print(type(request.tensor_content))
            # buffer.write(request.tensor_content)
            # binary_tensor = request.tensor_content
            input = torch.load(buffer)
            # print(request.tensor_content)
            with torch.no_grad():
                output = model(input, request.cp, model.layer_size)
            # temp_file = str(uuid.uuid1())
            print(output.size())
            buffer = io.BytesIO()
            torch.save(output, buffer)
            data = buffer.getvalue()
            # with codecs.open(buffer, 'rb') as f:
                # f.seek(0)
                # data = f.read()
            return model_split_pb2.ForwardResult(tensor_content=data,
                                                 code=200)
        except:
            return model_split_pb2.ForwardResult(pickle("error"), code=500)


def serve():
    # multi-thread
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    model_split_pb2_grpc.add_InferServicer_to_server(InferServicer(), server)
    server.add_insecure_port('[::]:10086')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()