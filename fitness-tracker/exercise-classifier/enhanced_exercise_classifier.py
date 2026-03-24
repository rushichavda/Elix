import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pickle
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

@dataclass
class ExerciseTemplate:
    """Template for ideal exercise form"""
    name: str
    keypoint_sequence: List[List[float]]  # Normalized keypoint positions over time
    angle_sequence: List[Dict[str, float]]  # Joint angles over time
    duration: float  # Duration in seconds
    quality_score: float  # Quality of this template (100 = perfect)

class TemplateManager:
    """Manages exercise templates for comparison"""
    
    def __init__(self):
        self.templates = {}
        self.template_dir = "templates"
        os.makedirs(self.template_dir, exist_ok=True)
        
    def create_template(self, exercise_type, keypoint_sequence, angle_sequence, duration):
        """Create a new exercise template"""
        template = ExerciseTemplate(
            name=exercise_type,
            keypoint_sequence=keypoint_sequence,
            angle_sequence=angle_sequence,
            duration=duration,
            quality_score=100.0  # Assume template is perfect form
        )
        
        if exercise_type not in self.templates:
            self.templates[exercise_type] = []
        
        self.templates[exercise_type].append(template)
        self.save_template(template)
        return template
    
    def save_template(self, template):
        """Save template to disk"""
        filename = f"{self.template_dir}/{template.name}_template_{len(self.templates.get(template.name, []))}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(template, f)
    
    def load_templates(self):
        """Load all templates from disk"""
        if not os.path.exists(self.template_dir):
            return
        
        for filename in os.listdir(self.template_dir):
            if filename.endswith('.pkl'):
                try:
                    with open(os.path.join(self.template_dir, filename), 'rb') as f:
                        template = pickle.load(f)
                        if template.name not in self.templates:
                            self.templates[template.name] = []
                        self.templates[template.name].append(template)
                except:
                    print(f"Failed to load template: {filename}")
    
    def get_best_template(self, exercise_type):
        """Get the best template for an exercise type"""
        if exercise_type not in self.templates or not self.templates[exercise_type]:
            return None
        return max(self.templates[exercise_type], key=lambda t: t.quality_score)

class TemplateComparator:
    """Compare exercise performance with ideal templates using DTW"""
    
    def __init__(self):
        self.template_manager = TemplateManager()
        self.template_manager.load_templates()
        
    def normalize_keypoints(self, keypoints):
        """Normalize keypoints relative to body dimensions"""
        if not keypoints:
            return []
        
        # Use shoulder width for normalization
        left_shoulder = keypoints[11] if len(keypoints) > 11 else None
        right_shoulder = keypoints[12] if len(keypoints) > 12 else None
        
        if not left_shoulder or not right_shoulder:
            return []
        
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)
        if shoulder_width == 0:
            return []
        
        normalized = []
        center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        for landmark in keypoints:
            norm_x = (landmark.x - center_x) / shoulder_width
            norm_y = (landmark.y - center_y) / shoulder_width
            normalized.extend([norm_x, norm_y])
        
        return normalized
    
    def compare_with_template(self, angle_sequence, exercise_type):
        """Compare current exercise with ideal template using DTW"""
        template = self.template_manager.get_best_template(exercise_type)
        
        if not template or not angle_sequence:
            return 50.0, "No template available"
        
        try:
            # Prepare sequences for DTW comparison
            user_seq = []
            template_seq = []
            
            # Get common angles between user and template
            common_angles = set()
            for user_frame in angle_sequence:
                common_angles.update(user_frame.keys())
            
            if not template.angle_sequence:
                return 50.0, "Invalid template"
            
            for template_frame in template.angle_sequence:
                common_angles.intersection_update(template_frame.keys())
            
            if not common_angles:
                return 50.0, "No common angles found"
            
            # Build sequences
            for user_frame in angle_sequence:
                frame_data = [user_frame.get(angle, 0) for angle in sorted(common_angles)]
                user_seq.append(frame_data)
            
            for template_frame in template.angle_sequence:
                frame_data = [template_frame.get(angle, 0) for angle in sorted(common_angles)]
                template_seq.append(frame_data)
            
            if not user_seq or not template_seq:
                return 50.0, "Empty sequences"
            
            # Perform DTW comparison
            distance, path = fastdtw(user_seq, template_seq, dist=euclidean)
            
            # Convert distance to similarity score (0-100)
            max_possible_distance = 180 * len(common_angles) * max(len(user_seq), len(template_seq))
            normalized_distance = min(distance / max_possible_distance, 1.0)
            similarity_score = (1.0 - normalized_distance) * 100
            
            # Generate feedback based on similarity
            if similarity_score >= 90:
                feedback = "Excellent technique matching ideal form!"
            elif similarity_score >= 80:
                feedback = "Very good form, minor timing adjustments"
            elif similarity_score >= 70:
                feedback = "Good form, work on movement consistency"
            elif similarity_score >= 60:
                feedback = "Acceptable form, focus on proper technique"
            else:
                feedback = "Poor form, review proper technique"
            
            return similarity_score, feedback
            
        except Exception as e:
            print(f"Template comparison error: {e}")
            return 50.0, "Comparison failed"

class HybridClassifier:
    """Combines rule-based and template-based classification"""
    
    def __init__(self):
        self.rule_classifier = RuleBasedClassifier()
        self.template_comparator = TemplateComparator()
        
        # Weights for combining approaches
        self.weights = {
            'rule_based': 0.4,
            'template_matching': 0.6
        }
        
    def classify_exercise(self, angles, angle_sequence, exercise_type):
        """Hybrid classification combining rule-based and template matching"""
        
        # Rule-based analysis (real-time)
        rule_quality, rule_score, rule_feedback = self.rule_classifier.analyze_form(angles, exercise_type)
        
        # Template matching analysis (when we have enough data)
        template_score = 50.0  # Default score
        template_feedback = "Analyzing movement pattern..."
        
        if len(angle_sequence) >= 15:  # Need minimum sequence for DTW
            template_score, template_feedback = self.template_comparator.compare_with_template(
                angle_sequence[-15:], exercise_type  # Use last 15 frames
            )
        
        # Combine scores
        final_score = (rule_score * self.weights['rule_based'] + 
                      template_score * self.weights['template_matching'])
        
        # Determine overall quality
        if final_score >= 85:
            final_quality = "green"
            combined_feedback = f"Excellent! {rule_feedback}"
        elif final_score >= 70:
            final_quality = "orange" 
            combined_feedback = f"Good form. {template_feedback}"
        else:
            final_quality = "red"
            combined_feedback = f"Needs improvement. {rule_feedback}"
        
        return final_quality, final_score, combined_feedback, {
            'rule_score': rule_score,
            'template_score': template_score,
            'rule_feedback': rule_feedback,
            'template_feedback': template_feedback
        }

class RecordingMode:
    """Mode for recording ideal exercise templates"""
    
    def __init__(self, template_manager):
        self.template_manager = template_manager
        self.is_recording = False
        self.recorded_keypoints = []
        self.recorded_angles = []
        self.start_time = None
        
    def start_recording(self):
        """Start recording a new template"""
        self.is_recording = True
        self.recorded_keypoints = []
        self.recorded_angles = []
        self.start_time = time.time()
        print("Recording started... Perform ideal exercise form!")
        
    def stop_recording(self, exercise_type):
        """Stop recording and save template"""
        if not self.is_recording:
            return None
            
        self.is_recording = False
        duration = time.time() - self.start_time
        
        if len(self.recorded_angles) < 10:  # Need minimum data
            print("Recording too short, template not saved")
            return None
        
        template = self.template_manager.create_template(
            exercise_type, 
            self.recorded_keypoints, 
            self.recorded_angles, 
            duration
        )
        
        print(f"Template saved for {exercise_type}! Duration: {duration:.1f}s, Frames: {len(self.recorded_angles)}")
        return template
        
    def add_frame(self, keypoints, angles):
        """Add frame data to recording"""
        if self.is_recording:
            self.recorded_keypoints.append(keypoints)
            self.recorded_angles.append(angles)

# Import all classes from the original code
from exercise_form_classifier import (
    ExerciseConfig, AngleCalculator, ExerciseDetector, 
    RuleBasedClassifier, RepCounter, FeedbackSystem
)

class EnhancedExerciseFormApp:
    """Enhanced main application with template matching"""
    
    def __init__(self):
        self.detector = ExerciseDetector()
        self.hybrid_classifier = HybridClassifier()
        self.feedback_system = FeedbackSystem()
        self.rep_counter = None
        self.current_exercise = "squat"
        self.is_running = False
        
        # Template recording
        self.recording_mode = RecordingMode(self.hybrid_classifier.template_comparator.template_manager)
        self.angle_sequence = deque(maxlen=60)  # Store last 2 seconds at 30fps
        
        # UI colors
        self.colors = {
            'green': (0, 255, 0),
            'orange': (0, 165, 255),
            'red': (0, 0, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'blue': (255, 0, 0)
        }
        
    def set_exercise(self, exercise_type):
        """Set the current exercise type"""
        valid_exercises = ["squat", "pushup", "bicep_curl", "plank", "lunge"]
        if exercise_type in valid_exercises:
            self.current_exercise = exercise_type
            self.rep_counter = RepCounter(exercise_type)
            self.angle_sequence.clear()
            print(f"Exercise set to: {exercise_type}")
        else:
            print(f"Unknown exercise: {exercise_type}")
    
    def draw_enhanced_ui(self, image, quality, score, feedback_text, angles, rep_count, details):
        """Draw enhanced feedback UI with hybrid analysis details"""
        height, width = image.shape[:2]
        
        # Main feedback panel
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 200), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Quality indicator
        color = self.colors[quality]
        cv2.circle(image, (50, 50), 30, color, -1)
        cv2.putText(image, quality.upper(), (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
        
        # Hybrid scores
        cv2.putText(image, f"Overall: {score:.1f}%", (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['white'], 2)
        cv2.putText(image, f"Rules: {details['rule_score']:.1f}%", (100, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
        cv2.putText(image, f"Template: {details['template_score']:.1f}%", (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
        
        # Exercise info
        cv2.putText(image, f"Exercise: {self.current_exercise.title()}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
        cv2.putText(image, f"Reps: {rep_count}", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 2)
        
        # Recording indicator
        if self.recording_mode.is_recording:
            cv2.putText(image, "RECORDING TEMPLATE", (width-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.circle(image, (width-30, 30), 10, (0, 0, 255), -1)
        
        # Feedback text
        cv2.putText(image, feedback_text[:50], (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if len(feedback_text) > 50:
            cv2.putText(image, feedback_text[50:100], (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Angle information (right side)
        angle_x = width - 300
        cv2.putText(image, "Joint Angles:", (angle_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
        for i, (angle_name, angle_value) in enumerate(angles.items()):
            y_pos = 50 + (i * 20)
            cv2.putText(image, f"{angle_name}: {angle_value:.1f}°", (angle_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['white'], 1)
        
        # Controls help
        help_y = height - 120
        cv2.rectangle(overlay, (10, help_y), (width-10, height-10), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        controls = [
            "Controls: 1-5=Exercise | R=Reset | T=Record Template | S=Stop Recording | Q=Quit",
            "Position: 2-4m from camera | Side view for squats/lunges | Face camera for curls"
        ]
        
        for i, text in enumerate(controls):
            cv2.putText(image, text, (15, help_y + 25 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['white'], 1)
    
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
        
        print("Enhanced Exercise Form Classifier Started!")
        print("Features: Hybrid analysis (Rule-based + Template matching)")
        print("\nControls:")
        print("1-5: Switch exercises (1=Squat, 2=Push-up, 3=Bicep Curl, 4=Plank, 5=Lunge)")
        print("R: Reset rep counter")
        print("T: Start recording ideal template")
        print("S: Stop recording template")
        print("Q: Quit")
        print("\nTemplate Recording:")
        print("- Press T to start recording an ideal exercise")
        print("- Perform the exercise with perfect form")
        print("- Press S to stop and save the template")
        print("- Templates improve accuracy over time")
        
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
                    # Add to angle sequence for template matching
                    self.angle_sequence.append(angles.copy())
                    
                    # Recording mode
                    if self.recording_mode.is_recording:
                        normalized_keypoints = self.hybrid_classifier.template_comparator.normalize_keypoints(
                            results.pose_landmarks.landmark
                        )
                        self.recording_mode.add_frame(normalized_keypoints, angles)
                    
                    # Hybrid analysis
                    quality, score, feedback_text, details = self.hybrid_classifier.classify_exercise(
                        angles, list(self.angle_sequence), self.current_exercise
                    )
                    
                    # Update rep counter
                    if self.rep_counter:
                        self.rep_counter.update(angles, score)
                    
                    # Draw enhanced UI
                    rep_count = self.rep_counter.rep_count if self.rep_counter else 0
                    self.draw_enhanced_ui(frame, quality, score, feedback_text, angles, rep_count, details)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                print(f"FPS: {fps:.1f}")
            
            # Display frame
            cv2.imshow('Enhanced Exercise Form Classifier', frame)
            
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
            elif key == ord('t'):
                waiting_time = 3
                print(f"Starting template recording in {waiting_time} seconds...")
                time.sleep(waiting_time)
                self.recording_mode.start_recording()
            elif key == ord('s'):
                self.recording_mode.stop_recording(self.current_exercise)
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the enhanced application"""
    try:
        app = EnhancedExerciseFormApp()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()