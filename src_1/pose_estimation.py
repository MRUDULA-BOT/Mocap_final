import cv2
import mediapipe as mp
import numpy as np
import logging

class PoseEstimator:
    def __init__(self, 
                 pose_confidence=0.7, 
                 hands_confidence=0.7, 
                 face_confidence=0.7):
        print("[POSE] Initializing MediaPipe modules")
        
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=pose_confidence, 
            min_tracking_confidence=pose_confidence
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=hands_confidence,
            min_tracking_confidence=hands_confidence
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=face_confidence,
            min_tracking_confidence=face_confidence
        )
        
        print("[POSE] MediaPipe modules initialized successfully")
    
    def estimate_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        landmarks_list = []
        if results.pose_landmarks:
            print("[POSE] Pose landmarks detected")
            landmarks_list.append(results.pose_landmarks)
        return landmarks_list
    
    def estimate_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            print(f"[HANDS] {len(results.multi_hand_landmarks)} hand(s) detected")
        return results.multi_hand_landmarks
    
    def estimate_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            print(f"[FACE] {len(results.multi_face_landmarks)} face(s) detected")
        return results.multi_face_landmarks
    
    def draw_pose(self, frame, landmarks_list):
        if landmarks_list:
            for landmarks in landmarks_list:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                )
        return frame
    
    def draw_hands(self, frame, hands_landmarks):
        if hands_landmarks:
            for hand_landmarks in hands_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2)
                )
        return frame
    
    def draw_faces(self, frame, faces_landmarks):
        if faces_landmarks:
            for face_landmarks in faces_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    face_landmarks, 
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    self.mp_drawing.DrawingSpec(color=(255,255,0), thickness=1),
                    self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
                )
        return frame
