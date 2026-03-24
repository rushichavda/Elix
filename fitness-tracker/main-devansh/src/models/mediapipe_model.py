# Python file
# src/models/mediapipe_model.py
import mediapipe as mp
import numpy as np
import cv2
import time
from typing import Dict
from .base_model import BasePoseModel

class MediaPipeModel(BasePoseModel):
    """MediaPipe Pose estimation model wrapper"""
    
    def __init__(self, 
                 complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        super().__init__("MediaPipe")
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.load_model()
    
    def load_model(self):
        """Initialize MediaPipe Pose model"""
        self.model = self.mp_pose.Pose(
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process frame with MediaPipe"""
        start_time = time.time()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process with MediaPipe
        results = self.model.process(rgb_frame)
        
        # Extract keypoints
        keypoints = np.zeros((1, 33, 3))  # MediaPipe has 33 landmarks
        
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[0, idx] = [
                    landmark.x * frame.shape[1],  # Convert to pixel coordinates
                    landmark.y * frame.shape[0],
                    landmark.visibility  # MediaPipe uses visibility instead of confidence
                ]
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return {
            'keypoints': keypoints,
            'processing_time': processing_time,
            'frame_dimensions': (frame.shape[0], frame.shape[1]),
            'world_landmarks': results.pose_world_landmarks,  # 3D landmarks
            'segmentation_mask': results.segmentation_mask if self.enable_segmentation else None
        }

    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Draw keypoints using base class method for consistency across all models"""
        # Convert MediaPipe 33 keypoints to COCO 17 format for consistent visualization
        if keypoints.shape[1] == 33:  # MediaPipe format
            coco_keypoints = self.mediapipe_to_coco_format(keypoints)
        else:
            coco_keypoints = keypoints

        # Use base class drawing method
        annotated = super().draw_keypoints(frame, coco_keypoints)

        # Add MediaPipe-specific FPS info
        if self.processing_times:
            avg_time = np.mean(self.processing_times[-30:])
            fps = 1.0 / avg_time if avg_time > 0 else 0
            # Overwrite the base FPS text with MediaPipe-specific label
            cv2.putText(annotated, f"MediaPipe FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated
    
    def get_3d_landmarks(self, results):
        """Extract 3D world coordinates from MediaPipe results"""
        if results.pose_world_landmarks:
            landmarks_3d = np.zeros((33, 3))
            for idx, landmark in enumerate(results.pose_world_landmarks.landmark):
                landmarks_3d[idx] = [landmark.x, landmark.y, landmark.z]
            return landmarks_3d
        return None

    @staticmethod
    def mediapipe_to_coco_format(mediapipe_keypoints: np.ndarray) -> np.ndarray:
        """Convert MediaPipe 33 keypoints to COCO 17 keypoints format"""
        # Mapping from COCO index to MediaPipe index
        # Format: {coco_index: mediapipe_index}
        mapping = {
            0: 0,  # nose
            1: 2,  # left_eye
            2: 5,  # right_eye
            3: 7,  # left_ear
            4: 8,  # right_ear
            5: 11,  # left_shoulder
            6: 12,  # right_shoulder
            7: 13,  # left_elbow
            8: 14,  # right_elbow
            9: 15,  # left_wrist
            10: 16,  # right_wrist
            11: 23,  # left_hip
            12: 24,  # right_hip
            13: 25,  # left_knee
            14: 26,  # right_knee
            15: 27,  # left_ankle
            16: 28  # right_ankle
        }

        coco_keypoints = np.zeros((mediapipe_keypoints.shape[0], 17, 3))
        for coco_idx, mp_idx in mapping.items():
            if mp_idx < mediapipe_keypoints.shape[1]:
                coco_keypoints[:, coco_idx] = mediapipe_keypoints[:, mp_idx]

        return coco_keypoints