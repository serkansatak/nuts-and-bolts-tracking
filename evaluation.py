import trackeval
import json
import csv
import os
import motmetrics as mm

def format_for_mot(annotations_file: str, detections_file: str):
    
    with open(annotations_file, "r") as f:
        annotations = json.load(f)["annotations"]
        f.close()
        
    with open(detections_file, "r") as f:
        detections = json.load(f)["annotations"]
        f.close()
        
    annotations_str = str()
    detections_str = str()
        
    for anot in annotations:
        annotations_str += f"{anot['image_id']}, {anot['track_id']}, {anot['bbox'][0]}, {anot['bbox'][1]}, {anot['bbox'][2]}, {anot['bbox'][3]}, 1, -1, -1, -1" + os.linesep
        
    for det in detections:
        detections_str += f"{det['image_id']}, {det['track_id']}, {det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}, {det['confidence']}, -1, -1, -1" + os.linesep
        
    with open("mot_annotations.txt", "w") as f:
        f.write(annotations_str)
        f.close()
        
    with open("mot_detections.txt", "w") as f:
        f.write(detections_str)
        f.close()     
            
def motMetricsEnhancedCalculator(gtSource, tSource):
    
    """
    From https://github.com/cheind/py-motmetrics readme file.
    """

        
    # import required packages
    import motmetrics as mm
    import numpy as np
    
    # load ground truth
    gt = np.loadtxt(gtSource, delimiter=',')

    # load tracking output
    t = np.loadtxt(tSource, delimiter=',')

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:,0].max())):
        frame += 1 # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
        t_dets = t[t[:,0]==frame,1:6] # select all detections in t

        C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                    max_iou=0.5) # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:,0].astype('int').tolist(), \
                t_dets[:,0].astype('int').tolist(), C)

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                        'recall', 'precision', 'num_objects', \
                                        'mostly_tracked', 'partially_tracked', \
                                        'mostly_lost', 'num_false_positives', \
                                        'num_misses', 'num_switches', \
                                        'num_fragmentations', 'mota', 'motp' \
                                        ], \
                        name='acc')

    strsummary = mm.io.render_summary(
        summary,
        #formatters={'mota' : '{:.2%}'.format},
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
                'precision': 'Prcn', 'num_objects': 'GT', \
                'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
                'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
                'num_misses': 'FN', 'num_switches' : 'IDsw', \
                'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
                }
    )
    print(strsummary)
    
def evaluate_mot(annotations_file: str, detections_file: str):
    format_for_mot(annotations_file, detections_file)
    motMetricsEnhancedCalculator("./mot_annotations.txt", "./mot_detections.txt")