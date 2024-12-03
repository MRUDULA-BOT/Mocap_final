import cv2
import logging
from datetime import datetime

class CameraCapture:
    def __init__(self, camera_id=0, width=820, height=540):
        print(f"[CAMERA] Initializing camera with ID: {camera_id}")
        self.camera = cv2.VideoCapture(camera_id)
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        if not self.camera.isOpened():
            logging.error(f"[CAMERA] Unable to open camera {camera_id}")
            raise ValueError(f"Unable to open camera {camera_id}")
        
        print(f"[CAMERA] Successfully initialized camera")
    
    def get_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            print("[CAMERA] Failed to capture frame")
            return None
        return frame
    
    def release(self):
        print("[CAMERA] Releasing camera resources")
        self.camera.release()
    
    def __del__(self):
        self.release()
