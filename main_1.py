import cv2
import os
import time
import queue
import threading
from datetime import datetime

from src_1.camera_capture import CameraCapture
from src_1.pose_estimation import PoseEstimator
from src_1.data_processing import DataProcessor
from src_1.action_recognition import ActionRecognizer

class MotionCaptureApp:
    def __init__(self, width=620, height=540, max_queue_size=100):
        self.camera = CameraCapture(width=width, height=height)
        self.pose_estimator = PoseEstimator()
        self.data_processor = DataProcessor()
        self.action_recognizer = ActionRecognizer()
        
        self.landmark_queue = queue.Queue(maxsize=max_queue_size)
        self.is_running = True
        
        # Video writer setup
        output_dir = self.data_processor.output_dir
        video_filename = os.path.join(
            output_dir, 
            'videos', 
            f'motion_capture_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi'
        )
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(video_filename, self.fourcc, 20.0, (width, height))
        
        print(f"[MAIN] Video will be saved to {video_filename}")
    
    def landmark_processing_thread(self):
        while self.is_running:
            try:
                landmark_type, landmarks_np = self.landmark_queue.get(timeout=1)
                if landmarks_np is not None:
                    self.data_processor.save_landmarks_to_csv(landmarks_np, landmark_type)
                    self.data_processor.visualize_landmarks(landmarks_np, landmark_type)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[THREAD] Error processing landmarks: {e}")
    
    def run(self):
        # Start landmark processing thread
        processing_thread = threading.Thread(target=self.landmark_processing_thread)
        processing_thread.start()
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.is_running:
                frame = self.camera.get_frame()
                if frame is None:
                    break
                
                # Pose estimation
                pose_landmarks = self.pose_estimator.estimate_pose(frame)
                hand_landmarks = self.pose_estimator.estimate_hands(frame)
                face_landmarks = self.pose_estimator.estimate_face(frame)
                
                # Draw landmarks
                if pose_landmarks:
                    frame = self.pose_estimator.draw_pose(frame, pose_landmarks)
                    
                    # Action recognition
                    detected_action = self.action_recognizer.recognize_action(pose_landmarks)
                    if detected_action:
                        cv2.putText(frame, f"Action: {detected_action}", 
                                    (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 0), 2)
                
                if hand_landmarks:
                    frame = self.pose_estimator.draw_hands(frame, hand_landmarks)
                
                if face_landmarks:
                    frame = self.pose_estimator.draw_faces(frame, face_landmarks)
                
                # Save frame to video
                self.video_writer.write(frame)
                
                # Non-blocking landmark queue processing
                try:
                    if pose_landmarks:
                        pose_np = self.data_processor.landmarks_to_np(pose_landmarks)
                        self.landmark_queue.put_nowait(('pose', pose_np))
                    
                    if hand_landmarks:
                        hands_np = self.data_processor.landmarks_to_np(hand_landmarks)
                        self.landmark_queue.put_nowait(('hand', hands_np))
                    
                    if face_landmarks:
                        faces_np = self.data_processor.landmarks_to_np(face_landmarks)
                        self.landmark_queue.put_nowait(('face', faces_np))
                
                except queue.Full:
                    print("[MAIN] Landmark queue full, skipping frame")
                
                # Display frame
                cv2.imshow('Motion Capture', frame)
                
                # Break loop on 'q' key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # ESC key
                    break
                
                frame_count += 1
        
        except Exception as e:
            print(f"[MAIN] An error occurred: {e}")
        
        finally:
            # Stop threads and cleanup
            self.is_running = False
            processing_thread.join()
            
            self.camera.release()
            self.video_writer.release()
            cv2.destroyAllWindows()
            
            # Calculate and print FPS
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            print(f"[MAIN] Processed {frame_count} frames")
            print(f"[MAIN] Average FPS: {fps:.2f}")

def main():
    app = MotionCaptureApp()
    app.run()

if __name__ == "__main__":
    main()
