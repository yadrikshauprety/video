import cv2
from scenedetect import detect, ContentDetector

def extract_frames(video_path):
    # Detect scenes using Content-Aware Detection
    scene_list = detect(video_path, ContentDetector(threshold=27.0))
    cap = cv2.VideoCapture(video_path)
    frames = []

    if scene_list:
        for scene in scene_list:
            start_time_ms = scene[0].get_seconds() * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
    else:
        # Fallback to middle frame if no distinct scenes detected
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames