import rpctest_pb2_grpc
import rpctest_pb2
import grpc
import logging

def test_login(stub, uid, passwd):
    req = rpctest_pb2.LoginRequest(username=uid, passwd=passwd)
    response = stub.login(req)
    if response.responseCode == 200:
        print("okkkk")

def run():
    with grpc.insecure_channel('localhost:10086') as channel:
        stub = rpctest_pb2_grpc.UserStub(channel)
        print("Now we send login request")
        test_login(stub, '123', '123')
        print("done.")


if __name__ == '__main__':
    logging.basicConfig()
    run()