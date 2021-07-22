# mainly for MNIST-alike model
# demo use MNIST and simple Convlutional neural network
# Date: 2021 - 07 - 20
# Meng_chen@bupt.edu.cn
# gRPC server


import os
import io
import grpc
import torch
import logging
import argparse
import pickle
import torch.nn as nn
import model_split_pb2
import model_split_pb2_grpc
import torch.tensor as tensor
from model_configuration import *
from torch.autograd.grad_mode import no_grad
from concurrent import futures
import logging
import pickle
from model_configuration import *
import io

class InferServicer(model_split_pb2_grpc.InferServicer):
    # implement
    def __init__(self) -> None:
        super().__init__()

    def infer_part(self, request, context):
        try:
            model = model_dict[request.model_id] # get model instance
            model_param_path = '/code/modelparam/' + request.model_id.upper() +'.pt' # params path
            manager.load_params(model, model_param_path) # load params
            buffer = io.BytesIO(request.tensor_content) # io bytes buffer for storing tensor
            input = torch.load(buffer) # loads tensor for inference
            with torch.no_grad():
                output = model(input, request.cp, model.layer_size) # infer
            # print(output.size())
            # buffer.flush()  # useless
            buffer = io.BytesIO()
            torch.save(output, buffer) # same as client
            data = buffer.getvalue()
            return model_split_pb2.ForwardResult(tensor_content=data, # response
                                                 code=200)
        except:
            return model_split_pb2.ForwardResult(pickle("error"), code=500)


def serve():
    # multi-thread
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=3)) # multi-thread
    model_split_pb2_grpc.add_InferServicer_to_server(InferServicer(), server) # add servicer
    server.add_insecure_port('[::]:10086') # listen port
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()