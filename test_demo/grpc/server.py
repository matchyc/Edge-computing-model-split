import rpctest_pb2_grpc
import rpctest_pb2
import grpc
from concurrent import futures
import logging

class UserServicer(rpctest_pb2_grpc.UserServicer):
    # implement
    def __init__(self) -> None:
        super().__init__()

    def login(self, request, context):
        if request.username == '123':
            msg = "connected!"
            code = 200
            if code == 200:
                return rpctest_pb2.APIResponse(
                    responsemessage = msg,
                    responseCode = code
                )
            else:
                return rpctest_pb2.APIResponse(
                    responsemessage = 'error',
                    responseCode = 500
                )

def serve():
    # multi-thread
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rpctest_pb2_grpc.add_UserServicer_to_server(UserServicer(),
    server)
    server.add_insecure_port('[::]:10086')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()