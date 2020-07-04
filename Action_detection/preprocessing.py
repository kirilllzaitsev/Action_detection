import numpy as np
import cv2
from pathlib import Path

def add_missing_frames(video: Path) -> None:
    vidcap = cv2.VideoCapture(str(video))
    write_dir = str(video.parent)
    success,image = vidcap.read()
    name, num = str(video).split('.')[0].split('_')[0], int(str(video).split('.')[0].split('_')[1])
    while success:
        cv2.imwrite(f'{name}_{num}.jpg', image)     # save frame as JPEG file      
        success,image = vidcap.read()
        num += 1