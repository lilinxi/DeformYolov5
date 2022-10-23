#python -m pip install -r requirements.txt
conda activate Yolov5
python server.py
#nohup python server.py &

sudo ln -s /home/lmf/anaconda3/envs/Yolov5/bin/python /usr/bin/python
sudo python server.py