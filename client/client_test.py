from __future__ import print_function

import logging

import grpc
import proto_gen.detect_pb2
import proto_gen.detect_pb2_grpc


# https://github.com/grpc/grpc/blob/master/examples/python/helloworld/greeter_client.py
def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = proto_gen.detect_pb2_grpc.DeformYolov5Stub(channel)
        response = stub.Detect(
            proto_gen.detect_pb2.YoloModelRequest(
                image_path="/Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/yolo/yolov5/data/images/bus.jpg",
                image_size=640,
                weights_path="/Users/bytedance/PycharmProjects/211110_PanoDetectionProtobuf/yolo/yolov5/yolov5n.pt",
                conf_thres=0.25,
                iou_thres=0.45,
            ),
        )
    print("Greeter client received: ")
    print(response)


if __name__ == '__main__':
    logging.basicConfig()
    run()
