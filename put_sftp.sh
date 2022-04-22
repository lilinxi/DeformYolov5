#! /bin/bash

cd $(dirname $0)
./push.sh
cd ..
rm -rf ./DeformYolov5Sftp
git clone git@github.com:lilinxi/DeformYolov5.git DeformYolov5Sftp
sftp lab << EOF
put -r ./DeformYolov5Sftp /home/mm
EOF
