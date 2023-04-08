#!/bin/bash

BASEDIR=$(realpath $(dirname $0))

PRE_WEIGHTS="${BASEDIR}/weights/last.pt"
IM_SIZE=640
EPOCH_NUM=10
BATCH_NUM=5
PROJECT_DIR="${BASEDIR}"
DATA_YAML="${BASEDIR}/NutsBolts.yaml"
HYP_FILE="${BASEDIR}/hyp.finetune.yaml"

for arg in "$@";
do 
    case $arg in 
        --weights)
        PRE_WEIGHTS="$2"
        shift
        shift
        ;;

        --img)
        IM_SIZE="$2"
        shift
        shift
        ;;

        --epochs)
        EPOCH_NUM="$2"
        shift
        shift
        ;;

        --data)
        DATA_YAML="$2"
        shift
        shift
        ;;

        --batch)
        BATCH_NUM="$2"
        shift
        shift
        ;;

        --project)
        PROJECT_DIR="$2"
        shift
        shift
        ;;

        --batch)
        DATA_YAML="$2"
        shift
        shift
        ;;

        --hyp)
        HYP_FILE="$2"
        shift
        shift
        ;;
    esac
done

TRAIN_FILE="${PROJECT_DIR}/yolov5/train.py"

echo -e "
Parameters set for training...

weights         : ${PRE_WEIGHTS}
img             : ${IM_SIZE}
epoch           : ${EPOCH_NUM}
batch           : ${BATCH_NUM}
data            : ${DATA_YAML}
project         : ${PROJECT_DIR}
train_file      : ${TRAIN_FILE}
hyperparameters : ${HYP_FILE}
python path     : $(which python)
"

mkdir -p "${PROJECT_DIR}/runs"
PROJECT_DIR="${PROJECT_DIR}/runs"


while [ ! -f "$TRAIN_FILE" ]; do
    read -p "Unable to locate YOLOv5, do you wish to clone the repo? (y/n)  " yn
    case $yn in
        Yes | y | Y ) echo -e "\n"; git clone https://github.com/ultralytics/yolov5.git "${BASEDIR}/yolov5";;
        No | N | n ) echo -e "\nExiting, unable to train without YOLOv5...\n";exit;;
        *) echo -e "\nPlease answer y(yes) or n(no).\n";;
    esac
done

IS_CUDA=$(python -c "import torch; a=int(torch.cuda.is_available()); print(a)")
echo ${IS_CUDA}

if [ ${IS_CUDA} = 1 ]; then
    device=0;
else
    device="cpu";
fi

python ${TRAIN_FILE} \
    --img ${IM_SIZE} \
    --batch ${BATCH_NUM} \
    --epochs ${EPOCH_NUM} \
    --data ${DATA_YAML} \
    --weights ${PRE_WEIGHTS} \
    --project ${PROJECT_DIR} \
    --hyp ${HYP_FILE} \
    --device ${device} \
    --freeze 10
