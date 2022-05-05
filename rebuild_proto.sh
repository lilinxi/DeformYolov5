#!/usr/bin/env bash

#/usr/local/bin/python3 -m pip install grpcio
#/usr/local/bin/python3 -m pip install grpcio-tools

rm -rf proto_gen
mkdir proto_gen
/usr/local/bin/python3 -m grpc_tools.protoc -I ./proto --python_out=proto_gen --grpc_python_out=proto_gen detect.proto