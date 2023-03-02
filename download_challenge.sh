#!/bin/bash
BASEDIR=$(realpath $(dirname $0))

$(wget https://github.com/Stroma-Vision/machine-learning-challenge/releases/download/v0.1/challenge.zip)
$(unzip "./challenge.zip" -d ${BASEDIR})
$(rm -f "./challenge.zip")

$(mkdir -p ${BASEDIR}/dataset/labels/train ${BASEDIR}/dataset/labels/test ${BASEDIR}/dataset/labels/val \
    ${BASEDIR}/dataset/images/val ${BASEDIR}/dataset/images/train ${BASEDIR}/dataset/images/test)

$(python extract_frames.py)
$(python coco2yolo.py -j ./challenge/annotations/instances_train.json -o ./dataset/labels/train)
$(python coco2yolo.py -j ./challenge/annotations/instances_val.json -o ./dataset/labels/val)
$(python coco2yolo.py -j ./challenge/annotations/instances_test.json -o ./dataset/labels/test)