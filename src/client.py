# mainly for MNIST-alike model
# demo use MNIST and simple Convlutional neural network
# Date: 2021 - 07 - 20
# Meng_chen@bupt.edu.cn
# gRPC client

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
from model_configuration import *
import argparse
import uuid
import codecs
import os
import io

parser = argparse.ArgumentParser('传入参数：***.py')
parser.add_argument('-n','--name', default='mnist')

args = parser.parse_args()

model_id = args.name
model = model_dict[args.name]
model_param_path = '../modelparam/' + model_id.upper() +'.pt'


def front_end_infer(cp):
    with torch.no_grad():
        output = model(manager.input_data, 0,int(cp))
    return output
    
def send_to_server(stub, cp, content):
    # message PartitionInfo {
    # string model_id = 1;    // indicate the model
    # bytes tensor_content = 2; // tensor data
    # int32 cp = 3;   // cut point
    req = model_split_pb2.PartitionInfo(
        model_id=model_id,
        tensor_content=content,
        cp=cp
    )
    
    response = stub.infer_part(req)
    if response.code == 200:
        buffer = io.BytesIO(response.tensor_content)
        output = torch.load(buffer)
        print(output.size())
        pred = output.max(1, keepdim=True)[1]
        # print(pred)
        correct = pred.eq(manager.label.view_as(pred)).sum().item()
        print("Correct count:{}".format(correct))
    elif response.code == 500:
        print("server internal error")


def run():
    manager.prepare_data(model_id)
    dataiter = iter(manager.test_loader)
    manager.input_data, manager.label = dataiter.next()
    manager.load_params(model, model_param_path)

    with grpc.insecure_channel('localhost:10087') as channel:
        stub = model_split_pb2_grpc.InferStub(channel)
        print("Now we begin front end inference...")
        output = front_end_infer(5) # hard-code
        buffer = io.BytesIO()
        torch.save(output, buffer)
        data = buffer.getvalue()
        print("Begin to send intermediate results...")
        send_to_server(stub, 5, data)

if __name__ == '__main__':
    logging.basicConfig()
    run()