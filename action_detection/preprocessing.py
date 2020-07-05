from pathlib import Path

import cv2


def add_missing_frames(video: Path) -> None:
    """
    Fill in missing images for action categories where
    sole video is provided
    """
    vidcap = cv2.VideoCapture(str(video))
    success, image = vidcap.read()
    name, num = str(video).split('.')[0].split('_')[0], int(str(video).split('.')[0].split('_')[1])
    while success:
        cv2.imwrite(f'{name}_{num}.jpg', image)
        success, image = vidcap.read()
        num += 1
