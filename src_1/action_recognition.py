import mediapipe as mp
import numpy as np

class ActionRecognizer:
    def __init__(self, confidence_threshold=0.5):
        self.mp_pose = mp.solutions.pose
        self.confidence_threshold = confidence_threshold
        self.actions = {
            'wave': self._detect_wave,
            'hello': self._detect_hello,
            'raise_hand': self._detect_raise_hand,
            'cross_arms': self._detect_cross_arms,
            'jump': self._detect_jump
        }
    
    def recognize_action(self, landmarks):
        if not landmarks:
            return None
        
        landmarks = landmarks[0].landmark
        for action_name, action_method in self.actions.items():
            if action_method(landmarks):
                return action_name
        
        return None
    
    def _detect_wave(self, landmarks):
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        return (right_wrist.y < right_shoulder.y and 
                abs(right_wrist.x - right_shoulder.x) > 0.2)
    
    def _detect_hello(self, landmarks):
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        
        return (right_wrist.y < right_ear.y and 
                abs(right_wrist.x - right_ear.x) < 0.2)
    
    def _detect_raise_hand(self, landmarks):
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        head_top = landmarks[self.mp_pose.PoseLandmark.NOSE]
        
        return right_wrist.y < head_top.y
    
    def _detect_cross_arms(self, landmarks):
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        center_x = landmarks[self.mp_pose.PoseLandmark.NOSE].x
        
        return (abs(right_wrist.x - center_x) < 0.2 and 
                abs(left_wrist.x - center_x) < 0.2)
    
    def _detect_jump(self, landmarks):
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        hip_level = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y
        
        return left_ankle.y < hip_level and right_ankle.y < hip_level
