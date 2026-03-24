# src/models/base_model.py
from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import time

class BasePoseModel(ABC):
    """Abstract base class for pose estimation models"""
    
    # COCO keypoint format (17 keypoints)
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.processing_times = []
        self.frame_count = 0
        
    @abstractmethod
    def load_model(self) -> Any:
        """Load the pose estimation model"""
        pass
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and return pose keypoints
        
        Returns:
            Dict containing:
            - keypoints: np.array of shape (n_people, n_keypoints, 3) [x, y, confidence]
            - processing_time: float
            - frame_dimensions: tuple (height, width)
        """
        pass
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """Process entire video and optionally save annotated output"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        results = []
        out = None
        
        if output_path:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            result = self.process_frame(frame)
            results.append(result)
            
            # Visualize if output requested
            if output_path and out is not None:
                annotated_frame = self.draw_keypoints(frame, result['keypoints'])
                out.write(annotated_frame)
            
            self.frame_count += 1
            
        cap.release()
        if out:
            out.release()
            
        return results

    # Better approach: Use different thresholds for keypoints vs skeleton
    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Draw keypoints and skeleton on frame"""
        annotated = frame.copy()

        # Define skeleton connections
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]

        # Separate thresholds for different models
        keypoint_threshold = 0.3
        skeleton_threshold = 0.25  # Lower threshold for skeleton connections

        for person_kpts in keypoints:
            # Draw keypoints
            for idx, (x, y, conf) in enumerate(person_kpts):
                if conf > keypoint_threshold:
                    cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(annotated, str(idx), (int(x), int(y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Draw skeleton with lower threshold
            for connection in skeleton:
                kpt1, kpt2 = connection
                if kpt1 <= len(person_kpts) and kpt2 <= len(person_kpts):
                    pt1 = person_kpts[kpt1 - 1]
                    pt2 = person_kpts[kpt2 - 1]
                    # Use lower threshold for skeleton connections
                    if pt1[2] > skeleton_threshold and pt2[2] > skeleton_threshold:
                        cv2.line(annotated,
                                 (int(pt1[0]), int(pt1[1])),
                                 (int(pt2[0]), int(pt2[1])),
                                 (0, 0, 255), 2)

        # Add FPS info
        if self.processing_times:
            avg_time = np.mean(self.processing_times[-30:])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated
    
    def get_performance_metrics(self) -> Dict:
        """Return performance metrics"""
        if not self.processing_times:
            return {}
            
        return {
            'model_name': self.model_name,
            'total_frames': self.frame_count,
            'avg_processing_time': np.mean(self.processing_times),
            'std_processing_time': np.std(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'avg_fps': 1.0 / np.mean(self.processing_times) if self.processing_times else 0
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.processing_times = []
        self.frame_count = 0