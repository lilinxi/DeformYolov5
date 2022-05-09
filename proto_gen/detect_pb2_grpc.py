# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import proto_gen.detect_pb2 as detect__pb2


class DeformYolov5Stub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Detect = channel.unary_unary(
            '/pano_detection.DeformYolov5/Detect',
            request_serializer=detect__pb2.YoloModelRequest.SerializeToString,
            response_deserializer=detect__pb2.YoloModelResponse.FromString,
        )


class DeformYolov5Servicer(object):
    """Missing associated documentation comment in .proto file."""

    def Detect(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DeformYolov5Servicer_to_server(servicer, server):
    rpc_method_handlers = {
        'Detect': grpc.unary_unary_rpc_method_handler(
            servicer.Detect,
            request_deserializer=detect__pb2.YoloModelRequest.FromString,
            response_serializer=detect__pb2.YoloModelResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'pano_detection.DeformYolov5', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class DeformYolov5(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Detect(request,
               target,
               options=(),
               channel_credentials=None,
               call_credentials=None,
               insecure=False,
               compression=None,
               wait_for_ready=None,
               timeout=None,
               metadata=None):
        return grpc.experimental.unary_unary(request, target, '/pano_detection.DeformYolov5/Detect',
                                             detect__pb2.YoloModelRequest.SerializeToString,
                                             detect__pb2.YoloModelResponse.FromString,
                                             options, channel_credentials,
                                             insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
