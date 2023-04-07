import argparse
import os
import torch
from time import time
import json

import numpy as np
import cv2
from similari import Sort, PositionalMetricType, BoundingBox # https://docs.rs/similari/latest/similari/

from collections import namedtuple
import sys

from torchvision.ops import nms, box_iou

from evaluation import evaluate_mot



ROOT = os.getcwd()
TEST_IMAGE = os.path.join(ROOT, "../dataset/images/val/0865.jpg")

IMSIZE = namedtuple('IMSIZE', ['W', 'H'])(640, 640)

CLASS_LABELS = {0: "Bolt", 1: "Nut"}
IMAGE_FORMATS = [".jpg", ".jpeg", ".png"]


def inference():
    
    """
    Load YOLOv5 with custom weights
    """
    
    detector = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights)
    detector.to('cuda')
    detector.float()
    detector.eval()
    
    
    """
    Initialize the tracker
    """
    tracker = Sort(shards=40, bbox_history=1, max_idle_epochs=35, method=PositionalMetricType.iou(threshold=args.iou))
    cap = cv2.VideoCapture(args.input)
    
    t0 = time()
    
    
    """
    Initialize the video writer
    """
    outVid = cv2.VideoWriter(os.path.join(VIDDIR, 'output.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, IMSIZE)
    
    frame_count = 0
    detection_count = 0
    
    detections_list = []
    
    while cap.isOpened():

        t1 = time()
        ret, frame = cap.read()
        
        if ret:
            """
            Clear detections with low confidence
            """
            detections = detector(frame).pred[0] 
            detections = detections[detections[:,4] > args.confidence]
            boxes = []
            
            
            """
            Get BBoxes for detections and predict active tracks with tracker
            """
            for (x1,y1,x2,y2,conf,cls) in detections:
                w, h = x2-x1, y2-y1
                box = BoundingBox(x1,y1,w,h).as_xyaah() # [x, y, angle, aspect, height]
                boxes.append((box, None))

            active_tracks = tracker.predict(boxes)
            idle_tracks = tracker.idle_tracks()
            #active_tracks.extend(idle_tracks)
            
            """
            Draw boxes and write ID's on frames.
            """
            
            detections = detections.cpu().float().numpy()
            
            for (conf,cls), p in zip(detections[:,-2:], active_tracks):
                #print(cls , p)
                box = p.predicted_bbox.as_ltwh() # [left, top, width, height]
                cv2.rectangle(frame,
                            pt1=(int(box.left), int(box.top)),
                            pt2=(int(box.left + box.width), int(box.top + box.height)),
                            color=(0,255,0), thickness=2)
                cv2.putText(frame, f'#{p.id} - {CLASS_LABELS[cls]} - {round(conf,1):.01}', (int(box.left), int(box.top)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                
                
                if args.evaluate:
                    detections_list.append(
                        {
                            "image_id": int(frame_count),
                            "category_id": int(cls) + 1,
                            "segmentation": [],
                            "bbox": [float(box.left), float(box.top), float(box.width), float(box.height)],
                            "area": float(box.width) * float(box.height),
                            "iscrowd": 0,
                            "id": detection_count,
                            "track_id": int(p.id),
                            "confidence": float(conf)
                        }
                    )
                
                detection_count += 1
            
            
            """
            Write out the video frame.
            """
            outVid.write(frame)
            #cv2.imwrite(os.path.join(IMDIR, f"{frame_count:04}.jpg"),  frame)            
            
            t2 = time()
            if t2-t1 > (1/30):
                print(f"Frame {frame_count} inference more than limit: {(t2-t1):.4f}")
            frame_count += 1
            
        else:
            break
        
    print(f"Average inference time per frame : {((time()-t0) / (frame_count+1)):.04f}")
    
    if args.evaluate:
        
        det_file = f"detections_{FILENAME}.json"
        
        with open(det_file, "w") as jfile:
            json.dump({"annotations": detections_list}, jfile, indent=4)
            jfile.close()
        
        evaluate_mot(args.annotations, det_file)
        
        


def non_max_suppression(detections: torch.Tensor, biased: bool = False):
    """
    parameters:
        detections -> np.ndarray[:,6] -> x1,x2,y1,y2,conf,cls
    returns:
        np.ndarray[:,6] -> x1,x2,y1,y2,conf,cls
    
    Summary:
        
    NMS might be class agnostic 
    However we can choose nuts over bolts for sanity.
    """
    
    det = detections.detach().clone()

    if biased:
        det[:,4][torch.where(det[:,-1] == 1)] *= args.nuts_bias

    print(detections[nms(det[:,:4], det[:,4], args.iou)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gst', action='store_true', help='GStreamer or Not')
    parser.add_argument('--weights', type=str, default=os.path.join(ROOT,'last.engine'), help='Detection weights.')
    parser.add_argument('--input', type=str, required=True, help='Input file')
    parser.add_argument('--confidence', type=float, default=0.65, help='Confidence score for detections.')
    parser.add_argument('--iou', type=float, default=0.3, help='IoU threshold for tracker.')
    parser.add_argument('--outdir', type=str, default="./detections", help='Output directory')
    parser.add_argument("--data-format", default="coco", choices=['coco', 'yolo'])
    parser.add_argument("--nuts-bias", type=float, default=1.05)
    parser.add_argument("--box_threshold", type=float, default=0.7)
    parser.add_argument("--evaluate", action='store_true', help= 'Evaluate detections vs annotations.')
    parser.add_argument("--annotations", type=str, help='Annotations file.')
    return parser.parse_args()



if __name__ == "__main__":
    
    global args
    args = parse_args()
    
    global IMDIR, VIDDIR, FILENAME
    
    FILENAME = os.path.splitext(os.path.split(args.input)[-1])[0]
    
    IMDIR = os.path.join(args.outdir, "images")
    VIDDIR = os.path.join(args.outdir, "video")
    
    os.makedirs(IMDIR, exist_ok=True)
    os.makedirs(VIDDIR, exist_ok=True)
    
    inference()