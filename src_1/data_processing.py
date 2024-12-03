import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class DataProcessor:
    def __init__(self, output_dir='motion_capture_output'):
        print("[DATA] Initializing DataProcessor")
        self.output_dir = output_dir
        self._create_output_directories()
    
    def _create_output_directories(self):
        base_dirs = [
            'pose_landmarks', 
            'hand_landmarks', 
            'face_landmarks', 
            'videos', 
            'visualizations',
            'csv_data'
        ]
        for subdir in base_dirs:
            full_path = os.path.join(self.output_dir, subdir)
            os.makedirs(full_path, exist_ok=True)
            print(f"[DATA] Created directory: {full_path}")
    
    def landmarks_to_np(self, landmarks_list):
        if not landmarks_list:
            print("[DATA] No landmarks to process")
            return None
        
        np_landmarks = []
        for landmarks in landmarks_list:
            landmark_coords = [[l.x, l.y, l.z] for l in landmarks.landmark]
            np_landmarks.append(np.array(landmark_coords))
        
        print(f"[DATA] Converted {len(np_landmarks)} landmark sets to numpy arrays")
        return np_landmarks
    
    def save_landmarks_to_csv(self, landmarks, landmark_type):
        if not landmarks:
            print(f"[DATA] No {landmark_type} landmarks to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.output_dir, 
            'csv_data', 
            f'{landmark_type}_landmarks_{timestamp}.csv'
        )
        
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Landmark_Index', 'X', 'Y', 'Z'])
            
            for landmarks_set in landmarks:
                for i, coords in enumerate(landmarks_set):
                    csvwriter.writerow([i, coords[0], coords[1], coords[2]])
        
        print(f"[DATA] {landmark_type} landmarks saved to {filename}")
    
    def visualize_landmarks(self, landmarks, landmark_type, filename=None):
        if not landmarks:
            print(f"[DATA] No {landmark_type} landmarks to visualize")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.output_dir, 
                'visualizations', 
                f'{landmark_type}_visualization_{timestamp}.png'
            )
        
        plt.figure(figsize=(15, 10))
        
        # 2D Scatter Plot
        plt.subplot(221)
        for landmarks_set in landmarks:
            plt.scatter(landmarks_set[:, 0], landmarks_set[:, 1], c='r', marker='o')
            plt.title(f'2D {landmark_type.capitalize()} Landmarks')
            plt.xlabel('X')
            plt.ylabel('Y')
        
        # 3D Scatter Plot
        plt.subplot(222, projection='3d')
        for landmarks_set in landmarks:
            plt.scatter(
                landmarks_set[:, 0], 
                landmarks_set[:, 1], 
                landmarks_set[:, 2], 
                c='b', marker='o'
            )
            plt.title(f'3D {landmark_type.capitalize()} Landmarks')
        
        # Landmark Distribution Histogram
        plt.subplot(223)
        all_landmarks = np.concatenate(landmarks)
        plt.hist(all_landmarks[:, 0], bins=20, alpha=0.5, label='X')
        plt.hist(all_landmarks[:, 1], bins=20, alpha=0.5, label='Y')
        plt.title('Landmark X & Y Distribution')
        plt.legend()
        
        # Heatmap of Landmark Connections
        plt.subplot(224)
        correlation_matrix = np.corrcoef(all_landmarks.T)
        plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
        plt.title('Landmark Correlation Heatmap')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        print(f"[DATA] {landmark_type} landmarks visualization saved to {filename}")
