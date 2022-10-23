from concurrent import futures
import logging

import grpc

import proto_gen.detect_pb2
import proto_gen.detect_pb2_grpc

import detect_cache_utils


# https://github.com/grpc/grpc/blob/master/examples/python/helloworld/greeter_server.py
class DeformYolov5Server(proto_gen.detect_pb2_grpc.DeformYolov5Servicer):
    def Detect(
            self,
            request: proto_gen.detect_pb2.YoloModelRequest,
            context
    ) -> proto_gen.detect_pb2.YoloModelResponse:
        return detect_cache_utils.CacheYolov5Detect(request)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proto_gen.detect_pb2_grpc.add_DeformYolov5Servicer_to_server(DeformYolov5Server(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    logging.info("DeformYolov5 waiting at 50051")
    serve()
