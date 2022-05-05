import os
import hashlib
import logging

import proto_gen.detect_pb2
import detect_rpc_utils


# 将 Req 序列化成哈希值
def HashYoloModelRequest(req: proto_gen.detect_pb2.YoloModelRequest) -> str:
    req_str = req.SerializeToString()
    req_md5 = hashlib.md5(req_str).hexdigest()
    with open(req.image_path, 'rb') as fp:
        data = fp.read()
    file_md5 = hashlib.md5(data).hexdigest()
    return f'{req_md5}_{file_md5}'


# 使用缓存，没有则调用函数，并将结果缓存
def CacheDetectDecorator(
        req: proto_gen.detect_pb2.YoloModelRequest,
        detect_func: callable,
        cache_dir: str = '/tmp/cache/pano_detect/yolov5_detect') -> proto_gen.detect_pb2.YoloModelResponse:
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    hash_req = HashYoloModelRequest(req)
    cache_file = f'{cache_dir}/{hash_req}.txt'
    if os.path.exists(cache_file):
        logging.info('cache in %s', cache_file)
        yolo_model_resp = proto_gen.detect_pb2.YoloModelResponse()
        yolo_model_resp.ParseFromString(open(cache_file, 'rb').read())
    else:
        logging.info('cache miss %s', cache_file)
        yolo_model_resp = detect_func(req)
        with open(cache_file, 'wb') as f:
            f.write(yolo_model_resp.SerializeToString())
    return yolo_model_resp


def CacheYolov5Detect(yolo_model_req: proto_gen.detect_pb2.YoloModelRequest) \
        -> proto_gen.detect_pb2.YoloModelResponse:
    yolo_model_resp = CacheDetectDecorator(
        yolo_model_req,
        detect_func=detect_rpc_utils.Yolov5Detect)
    return yolo_model_resp
