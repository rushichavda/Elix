import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json

@dataclass
class ExerciseConfig:
    """Configuration for each exercise type"""
    name: str
    key_angles: Dict[str, str]  # angle_name: description
    ideal_ranges: Dict[str, Tuple[float, float]]  # angle_name: (min, max)
    critical_joints: List[str]
    feedback_phrases: Dict[str, str]

class AngleCalculator:
    """Utility class for calculating angles between pose landmarks"""
    
    @staticmethod
    def calculate_angle(point1, point2, point3):
        """
        Calculate angle between three points
        point2 is the vertex of the angle
        """
        try:
            # Convert to numpy arrays
            a = np.array([point1.x, point1.y])
            b = np.array([point2.x, point2.y])
            c = np.array([point3.x, point3.y])
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Calculate angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Handle numerical errors
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
        except:
            return 0.0

    @staticmethod
    def get_landmark_coords(landmarks, landmark_idx):
        """Get landmark coordinates with error handling"""
        try:
            return landmarks.landmark[landmark_idx]
        except:
            return None

class ExerciseDetector:
    """Core pose detection and angle extraction"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.angle_calc = AngleCalculator()
        
    def detect_pose(self, image):
        """Detect pose landmarks in image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results
    
    def extract_exercise_angles(self, landmarks, exercise_type):
        """Extract relevant angles for specific exercise"""
        if not landmarks:
            return {}
            
        angles = {}
        
        if exercise_type == "squat":
            # Right side angles
            right_hip_angle = self.angle_calc.calculate_angle(
                self.angle_calc.get_landmark_coords(landmarks, 12),  # right shoulder
                self.angle_calc.get_landmark_coords(landmarks, 24),  # right hip
                self.angle_calc.get_landmark_coords(landmarks, 26)   # right knee
            )
            
            right_knee_angle = self.angle_calc.calculate_angle(
                self.angle_calc.get_landmark_coords(landmarks, 24),  # right hip
                self.angle_calc.get_landmark_coords(landmarks, 26),  # right knee
                self.angle_calc.get_landmark_coords(landmarks, 28)   # right ankle
            )
            
            # Spine angle (shoulder to hip to knee alignment)
            spine_angle = self.angle_calc.calculate_angle(
                self.angle_calc.get_landmark_coords(landmarks, 11),  # left shoulder
                self.angle_calc.get_landmark_coords(landmarks, 23),  # left hip
                self.angle_calc.get_landmark_coords(landmarks, 25)   # left knee
            )
            
            angles = {
                'right_hip': right_hip_angle,
                'right_knee': right_knee_angle,
                'spine': spine_angle
            }
            
        elif exercise_type == "pushup":
            # Right side angles
            right_elbow_angle = self.angle_calc.calculate_angle(
                self.angle_calc.get_landmark_coords(landmarks, 12),  # right shoulder
                self.angle_calc.get_landmark_coords(landmarks, 14),  # right elbow
                self.angle_calc.get_landmark_coords(landmarks, 16)   # right wrist
            )
            
            # Body alignment (shoulder to hip to ankle)
            body_angle = self.angle_calc.calculate_angle(
                self.angle_calc.get_landmark_coords(landmarks, 12),  # right shoulder
                self.angle_calc.get_landmark_coords(landmarks, 24),  # right hip
                self.angle_calc.get_landmark_coords(landmarks, 28)   # right ankle
            )
            
            angles = {
                'right_elbow': right_elbow_angle,
                'body_alignment': body_angle
            }
            
        elif exercise_type == "bicep_curl":
            # Right arm curl
            right_elbow_angle = self.angle_calc.calculate_angle(
                self.angle_calc.get_landmark_coords(landmarks, 12),  # right shoulder
                self.angle_calc.get_landmark_coords(landmarks, 14),  # right elbow
                self.angle_calc.get_landmark_coords(landmarks, 16)   # right wrist
            )
            
            # Shoulder stability (should remain relatively static)
            shoulder_angle = self.angle_calc.calculate_angle(
                self.angle_calc.get_landmark_coords(landmarks, 11),  # left shoulder
                self.angle_calc.get_landmark_coords(landmarks, 12),  # right shoulder
                self.angle_calc.get_landmark_coords(landmarks, 14)   # right elbow
            )
            
            angles = {
                'right_elbow': right_elbow_angle,
                'shoulder_stability': shoulder_angle
            }
            
        elif exercise_type == "plank":
            # Body line from shoulder to hip to ankle
            body_line = self.angle_calc.calculate_angle(
                self.angle_calc.get_landmark_coords(landmarks, 12),  # right shoulder
                self.angle_calc.get_landmark_coords(landmarks, 24),  # right hip
                self.angle_calc.get_landmark_coords(landmarks, 28)   # right ankle
            )
            
            # Hip position
            hip_angle = self.angle_calc.calculate_angle(
                self.angle_calc.get_landmark_coords(landmarks, 12),  # right shoulder
                self.angle_calc.get_landmark_coords(landmarks, 24),  # right hip
                self.angle_calc.get_landmark_coords(landmarks, 26)   # right knee
            )
            
            angles = {
                'body_line': body_line,
                'hip_position': hip_angle
            }
            
        elif exercise_type == "lunge":
            # Front knee angle (assuming right leg forward)
            front_knee_angle = self.angle_calc.calculate_angle(
                self.angle_calc.get_landmark_coords(landmarks, 24),  # right hip
                self.angle_calc.get_landmark_coords(landmarks, 26),  # right knee
                self.angle_calc.get_landmark_coords(landmarks, 28)   # right ankle
            )
            
            # Hip flexion
            hip_flexion = self.angle_calc.calculate_angle(
                self.angle_calc.get_landmark_coords(landmarks, 12),  # right shoulder
                self.angle_calc.get_landmark_coords(landmarks, 24),  # right hip
                self.angle_calc.get_landmark_coords(landmarks, 26)   # right knee
            )
            
            angles = {
                'front_knee': front_knee_angle,
                'hip_flexion': hip_flexion
            }
        
        return angles

class RuleBasedClassifier:
    """Rule-based form analysis using biomechanical standards"""
    
    def __init__(self):
        self.exercise_configs = {
            "squat": ExerciseConfig(
                name="squat",
                key_angles={"right_hip": "Hip flexion", "right_knee": "Knee flexion", "spine": "Spine alignment"},
                ideal_ranges={"right_hip": (70, 120), "right_knee": (90, 130), "spine": (160, 180)},
                critical_joints=["right_knee", "spine"],
                feedback_phrases={
                    "good": "Great squat form!",
                    "moderate": "Good depth, watch knee alignment",
                    "poor": "Focus on knee tracking and depth"
                }
            ),
            "pushup": ExerciseConfig(
                name="pushup",
                key_angles={"right_elbow": "Elbow flexion", "body_alignment": "Body line"},
                ideal_ranges={"right_elbow": (90, 160), "body_alignment": (170, 190)},
                critical_joints=["right_elbow", "body_alignment"],
                feedback_phrases={
                    "good": "Perfect push-up form!",
                    "moderate": "Good form, maintain body line",
                    "poor": "Focus on full range and body alignment"
                }
            ),
            "bicep_curl": ExerciseConfig(
                name="bicep_curl",
                key_angles={"right_elbow": "Elbow flexion", "shoulder_stability": "Shoulder position"},
                ideal_ranges={"right_elbow": (40, 160), "shoulder_stability": (80, 100)},
                critical_joints=["right_elbow"],
                feedback_phrases={
                    "good": "Excellent curl technique!",
                    "moderate": "Good range, control the movement",
                    "poor": "Full range needed, avoid momentum"
                }
            ),
            "plank": ExerciseConfig(
                name="plank",
                key_angles={"body_line": "Body alignment", "hip_position": "Hip position"},
                ideal_ranges={"body_line": (170, 190), "hip_position": (160, 190)},
                critical_joints=["body_line", "hip_position"],
                feedback_phrases={
                    "good": "Perfect plank position!",
                    "moderate": "Good stability, adjust hip height",
                    "poor": "Maintain straight body line"
                }
            ),
            "lunge": ExerciseConfig(
                name="lunge",
                key_angles={"front_knee": "Front knee angle", "hip_flexion": "Hip position"},
                ideal_ranges={"front_knee": (85, 95), "hip_flexion": (160, 190)},
                critical_joints=["front_knee"],
                feedback_phrases={
                    "good": "Excellent lunge form!",
                    "moderate": "Good depth, check knee position",
                    "poor": "Focus on 90-degree knee angle"
                }
            )
        }
    
    def analyze_form(self, angles, exercise_type):
        """Analyze form quality based on joint angles"""
        if exercise_type not in self.exercise_configs:
            return "unknown", 0.0, "Exercise not recognized"
        
        config = self.exercise_configs[exercise_type]
        total_score = 0
        angle_scores = {}
        
        for angle_name, angle_value in angles.items():
            if angle_name in config.ideal_ranges:
                min_val, max_val = config.ideal_ranges[angle_name]
                
                # Calculate score for this angle (0-100)
                if min_val <= angle_value <= max_val:
                    score = 100  # Perfect
                else:
                    # Deduct points based on deviation
                    if angle_value < min_val:
                        deviation = min_val - angle_value
                    else:
                        deviation = angle_value - max_val
                    
                    # More penalty for critical joints
                    penalty_multiplier = 2 if angle_name in config.critical_joints else 1
                    score = max(0, 100 - (deviation * penalty_multiplier))
                
                angle_scores[angle_name] = score
        
        # Calculate weighted average (critical joints have more weight)
        weighted_scores = []
        for angle_name, score in angle_scores.items():
            weight = 2 if angle_name in config.critical_joints else 1
            weighted_scores.extend([score] * weight)
        
        if weighted_scores:
            total_score = np.mean(weighted_scores)
        
        # Determine quality level
        if total_score >= 85:
            quality = "green"
            feedback = config.feedback_phrases["good"]
        elif total_score >= 70:
            quality = "orange"
            feedback = config.feedback_phrases["moderate"]
        else:
            quality = "red"
            feedback = config.feedback_phrases["poor"]
        
        return quality, total_score, feedback

class RepCounter:
    """Count repetitions and track exercise phases"""
    
    def __init__(self, exercise_type):
        self.exercise_type = exercise_type
        self.rep_count = 0
        self.current_phase = "up"  # up, down, transition
        self.angle_history = deque(maxlen=10)
        self.rep_scores = []
        
        # Define phase thresholds for each exercise
        self.phase_thresholds = {
            "squat": {"down_threshold": 100, "up_threshold": 140},
            "pushup": {"down_threshold": 110, "up_threshold": 150},
            "bicep_curl": {"down_threshold": 60, "up_threshold": 140},
            "lunge": {"down_threshold": 90, "up_threshold": 160}
        }
    
    def update(self, angles, form_score):
        """Update rep count based on key angle progression"""
        if self.exercise_type not in self.phase_thresholds:
            return
        
        # Get primary angle for rep counting
        primary_angle = self._get_primary_angle(angles)
        if primary_angle is None:
            return
        
        self.angle_history.append(primary_angle)
        
        if len(self.angle_history) < 3:
            return
        
        thresholds = self.phase_thresholds[self.exercise_type]
        
        # State machine for rep counting
        if self.current_phase == "up":
            if primary_angle < thresholds["down_threshold"]:
                self.current_phase = "down"
        
        elif self.current_phase == "down":
            if primary_angle > thresholds["up_threshold"]:
                self.current_phase = "up"
                self.rep_count += 1
                self.rep_scores.append(form_score)
    
    def _get_primary_angle(self, angles):
        """Get the primary angle used for rep counting"""
        primary_angles = {
            "squat": "right_knee",
            "pushup": "right_elbow", 
            "bicep_curl": "right_elbow",
            "lunge": "front_knee"
        }
        
        if self.exercise_type in primary_angles:
            angle_name = primary_angles[self.exercise_type]
            return angles.get(angle_name)
        return None
    
    def get_average_score(self):
        """Get average form score across all reps"""
        if not self.rep_scores:
            return 0
        return np.mean(self.rep_scores)

class FeedbackSystem:
    """Real-time feedback and coaching system"""
    
    def __init__(self):
        self.feedback_history = deque(maxlen=30)  # 1 second of feedback at 30fps
        self.last_feedback_time = time.time()
        self.feedback_interval = 1.5  # seconds
        
    def add_feedback(self, quality, score, feedback_text, angles):
        """Add new feedback data"""
        self.feedback_history.append({
            'quality': quality,
            'score': score,
            'feedback': feedback_text,
            'angles': angles,
            'timestamp': time.time()
        })
    
    def get_current_feedback(self):
        """Get smoothed current feedback"""
        if not self.feedback_history:
            return "red", 0, "No data available", {}
        
        # Get recent feedback
        recent_feedback = list(self.feedback_history)[-10:]  # Last 10 frames
        
        # Calculate average score
        avg_score = np.mean([f['score'] for f in recent_feedback])
        
        # Determine overall quality
        if avg_score >= 85:
            overall_quality = "green"
        elif avg_score >= 70:
            overall_quality = "orange"
        else:
            overall_quality = "red"
        
        # Get most recent feedback text
        latest_feedback = recent_feedback[-1]['feedback']
        latest_angles = recent_feedback[-1]['angles']
        
        return overall_quality, avg_score, latest_feedback, latest_angles
    
    def should_provide_feedback(self):
        """Check if enough time has passed to provide new feedback"""
        current_time = time.time()
        if current_time - self.last_feedback_time >= self.feedback_interval:
            self.last_feedback_time = current_time
            return True
        return False

class ExerciseFormApp:
    """Main application class"""
    
    def __init__(self):
        self.detector = ExerciseDetector()
        self.classifier = RuleBasedClassifier()
        self.feedback_system = FeedbackSystem()
        self.rep_counter = None
        self.current_exercise = "squat"
        self.is_running = False
        
        # UI colors
        self.colors = {
            'green': (0, 255, 0),
            'orange': (0, 165, 255),
            'red': (0, 0, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }
        
    def set_exercise(self, exercise_type):
        """Set the current exercise type"""
        if exercise_type in self.classifier.exercise_configs:
            self.current_exercise = exercise_type
            self.rep_counter = RepCounter(exercise_type)
            print(f"Exercise set to: {exercise_type}")
        else:
            print(f"Unknown exercise: {exercise_type}")
    
    def draw_landmarks(self, image, landmarks):
        """Draw pose landmarks on image"""
        if landmarks:
            self.detector.mp_drawing.draw_landmarks(
                image, 
                landmarks, 
                self.detector.mp_pose.POSE_CONNECTIONS,
                self.detector.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                self.detector.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
    
    def draw_feedback_ui(self, image, quality, score, feedback_text, angles, rep_count):
        """Draw feedback UI on image"""
        height, width = image.shape[:2]
        
        # Background for feedback
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 150), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Quality indicator
        color = self.colors[quality]
        cv2.circle(image, (50, 50), 30, color, -1)
        cv2.putText(image, quality.upper(), (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
        
        # Score
        cv2.putText(image, f"Score: {score:.1f}%", (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['white'], 2)
        
        # Exercise and rep count
        cv2.putText(image, f"Exercise: {self.current_exercise.title()}", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
        cv2.putText(image, f"Reps: {rep_count}", (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
        
        # Feedback text
        cv2.putText(image, feedback_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Angle information
        y_offset = 180
        cv2.putText(image, "Joint Angles:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
        for i, (angle_name, angle_value) in enumerate(angles.items()):
            y_pos = y_offset + 25 + (i * 20)
            cv2.putText(image, f"{angle_name}: {angle_value:.1f}°", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['white'], 1)
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Exercise Form Classifier Started!")
        print("Controls:")
        print("1-5: Switch exercises (1=Squat, 2=Push-up, 3=Bicep Curl, 4=Plank, 5=Lunge)")
        print("R: Reset rep counter")
        print("Q: Quit")
        
        self.is_running = True
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect pose
            results = self.detector.detect_pose(frame)
            
            if results.pose_landmarks:
                # Draw landmarks
                self.draw_landmarks(frame, results.pose_landmarks)
                
                # Extract angles for current exercise
                angles = self.detector.extract_exercise_angles(results.pose_landmarks, self.current_exercise)
                
                if angles:
                    # Analyze form
                    quality, score, feedback_text = self.classifier.analyze_form(angles, self.current_exercise)
                    
                    # Add to feedback system
                    self.feedback_system.add_feedback(quality, score, feedback_text, angles)
                    
                    # Update rep counter
                    if self.rep_counter:
                        self.rep_counter.update(angles, score)
                    
                    # Get smoothed feedback
                    smooth_quality, smooth_score, smooth_feedback, smooth_angles = self.feedback_system.get_current_feedback()
                    
                    # Draw UI
                    rep_count = self.rep_counter.rep_count if self.rep_counter else 0
                    self.draw_feedback_ui(frame, smooth_quality, smooth_score, smooth_feedback, smooth_angles, rep_count)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                print(f"FPS: {fps:.1f}")
            
            # Display frame
            cv2.imshow('Exercise Form Classifier', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                self.set_exercise("squat")
            elif key == ord('2'):
                self.set_exercise("pushup")
            elif key == ord('3'):
                self.set_exercise("bicep_curl")
            elif key == ord('4'):
                self.set_exercise("plank")
            elif key == ord('5'):
                self.set_exercise("lunge")
            elif key == ord('r'):
                if self.rep_counter:
                    self.rep_counter = RepCounter(self.current_exercise)
                    print("Rep counter reset")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the application"""
    try:
        app = ExerciseFormApp()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()