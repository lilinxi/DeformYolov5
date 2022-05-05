#! /bin/bash

cd $(dirname $0)
rm -rf ./deform_yolov5.egg-info
rm -rf ./dist
/usr/local/bin/python3 setup.py build
/usr/local/bin/python3 setup.py sdist
cd dist
tar -zxvf deform_yolov5-1.0.tar.gz
cd deform_yolov5-1.0
/usr/local/bin/python3 setup.py install
ll /usr/local/lib/python3.9/site-packages | grep deform

