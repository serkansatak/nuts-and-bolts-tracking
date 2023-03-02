import cv2
import os

def get_fps(cap): 
    return cap.get(cv2.CAP_PROP_FPS)


def write_frames(cap, frame_dir):
    
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir, exist_ok=False)
    frame_count = 0
    
    while cap.isOpened():   
        ret, frame = cap.read()
        if ret:
            frame_path = os.path.join(frame_dir, f'{frame_count:04}'+'.jpg')
            cv2.imwrite(frame_path, frame)
            frame_count += 1
        else:
            break
    

if __name__ == "__main__":
    
    dataset_pattern = "./dataset/images/pattern_"
    path_pattern = "./challenge/images/pattern_/pattern_.mp4"
    global VID_PATH
    
    for train_mode in ['val', 'test', 'train']:
    
        VID_PATH = path_pattern.replace('pattern_', train_mode)
        FRAME_DIR = dataset_pattern.replace('pattern_', train_mode)
        cap = cv2.VideoCapture(VID_PATH)
        print(f"{train_mode} FPS : {get_fps(cap)}")
        write_frames(cap, FRAME_DIR)    
    