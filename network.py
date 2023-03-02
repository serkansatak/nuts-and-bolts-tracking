import os
import cv2
import torch
import time
from sort import *

IMDIR = "./preds/"


t1 = time.time()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./exp5/weights/last.pt') #, autoshape=False)
model.to('cuda')
model.float()
model.eval()

savepath = os.path.join(os.getcwd(), "Detections")
cap = cv2.VideoCapture("../challenge/images/val/val.mp4")
tracker = Sort()


label_dict = {0: "Bolt", 1: "Nut"}
label_color = {0: (0,255,0), 1: (255,0,0)}
i=0






while(True):
    ret, im = cap.read()
    
    if ret:
        
        t2 = time.time()
        preds = model(im)
        t3 = time.time()
        
        infer_time = t3-t2
        
        #print(f"Inference time : {infer_time:.4f}")
            
        if infer_time>(1/30):
            print("Fakaaaaaaaaa")
            
        detections = preds.pred[0].cpu().float().numpy()
        new_dets = detections[detections[:,4] > 0.7]
        #print(detections)
        #print(new_dets)
        track_bbs_ids = tracker.update(new_dets)
        
        for j in range(len(track_bbs_ids.tolist())):
            coords = track_bbs_ids.tolist()[j]
            x1, x2, y1, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            trackid = int(coords[-1])
            name_idx = int(new_dets[j][-1])
            name = label_dict[name_idx]
            color = label_color[name_idx]
            cv2.rectangle(im, (x1,y1), (x2,y2), color, 2)
            cv2.putText(im, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imwrite(os.path.join(IMDIR, f"{i:04}.jpg"), im)
        i+=1
        
    else:
        break
    
exc_time = time.time()-t1

print(f"Execution Time : {exc_time:.4f}\nAverage exc time per frame : {(exc_time/(i+1)):.4f}")    
cap.release()
cv2.destroyAllWindows()
