# mainly for MNIST-alike model
# demo use MNIST and simple Convlutional neural network
# Date: 2021 - 07 - 20
# Meng_chen@bupt.edu.cn
# gRPC client


import os
import io
import grpc
import torch
import logging
import argparse
import torch.nn as nn
import model_split_pb2
import model_split_pb2_grpc
import torch.tensor as tensor
from model_configuration import *
from torch.autograd.grad_mode import no_grad
from apscheduler.schedulers.background import BlockingScheduler, BackgroundScheduler

parser = argparse.ArgumentParser()
parser.add_argument('-n','--name', default='mnist') # default: mnist
args = parser.parse_args()

model_id = args.name
model = model_dict[args.name] # get indicated model
model_param_path = '../modelparam/' + model_id.upper() +'.pt' # uniform path

server_list = ['localhost:10086', 'localhost:10087'] # add more

server_index = 0

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
    global server_index
    manager.prepare_data(model_id) # get data by manager
    dataiter = iter(manager.test_loader) # just get data nothing...
    manager.input_data, manager.label = dataiter.next() 
    manager.load_params(model, model_param_path) # load trained parameters

    with grpc.insecure_channel(server_list[server_index]) as channel: # server_index indicate the server for this request
        stub = model_split_pb2_grpc.InferStub(channel)
        print("Now we begin front end inference...")
        output = front_end_infer(5) # hard-code for now could be changed easily
        buffer = io.BytesIO() # io bytes buffer for temporarily store tensor
        torch.save(output, buffer)
        data = buffer.getvalue()
        print("Begin to send intermediate results...| Target server: {}".format(server_list[server_index]))
        send_to_server(stub, 5, data) # rpc
    server_index = (server_index + 1) % len(server_list) # update server_index

if __name__ == '__main__':
    scheduler = BlockingScheduler()
    scheduler.add_job(run, 'interval', seconds=3)
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C  '))
    logging.basicConfig()
    
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass