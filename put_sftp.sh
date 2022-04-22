#! /bin/bash

read -p "Do you want to put the file? (y/n) " answer

cd $(dirname $0)
./push.sh

cd ..
cd DeformYolov5Sftp
git pull origin deform_yolov5
cd ..

#rm -rf ./DeformYolov5Sftp
#git clone git@github.com:lilinxi/DeformYolov5.git DeformYolov5Sftp

#sftp lab << EOF
#put -r ./DeformYolov5Sftp /home/mm
#EOF

sftp lab724 << EOF
put -r ./DeformYolov5Sftp /home/lmf
EOF
