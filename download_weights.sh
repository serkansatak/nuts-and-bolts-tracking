#!/bin/bash

export PYTHONPATH="./yolov5"
echo $PYTHONPATH

mkdir -p ./weights/initials/


python - <<EOF
from utils.downloads import attempt_download

p5 = list('sm')  # P5 models
p6 = [f'{x}6' for x in p5]  # P6 models
cls = [f'{x}-cls' for x in p5]  # classification models
seg = [f'{x}-seg' for x in p5]  # classification models

for x in p5 + p6 + cls + seg:
    attempt_download(f'./weights/initials/yolov5{x}.pt')

EOF