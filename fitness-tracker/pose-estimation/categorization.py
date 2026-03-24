import mediapipe as mp
import numpy as np
import math
from collections import deque
import time

class SquatFormAnalyzer:
    def __init__(self):
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Squat analysis parameters
        self.position_history = deque(maxlen=30)  # Track hip height over time
        self.squat_state = "standing"  # "standing", "descending", "bottom", "ascending"
        self.rep_count = 0
        self.rep_scores = []
        self.current_rep_data = {}
        
        # Thresholds for squat detection
        self.descent_threshold = 0.05  # Minimum descent to register movement
        self.ascent_threshold = 0.03   # Minimum ascent to complete rep
        
        # Perfect squat ranges for front-facing analysis
        self.perfect_ranges = {
            'depth_percentage': (25, 40),     # Hip drop percentage from standing
            'knee_alignment': (0.85, 1.15),   # Knee-to-ankle alignment ratio
            'symmetry_score': (0.9, 1.1),    # Left-right symmetry ratio
            'hip_knee_ratio': (1.05, 1.25)   # Hip below knee level
        }
        
        # Track standing position for depth calculation
        self.standing_hip_height = None
        self.standing_knee_height = None
        self.calibration_frames = 0
        self.calibration_data = []
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        # Convert to numpy arrays
        p1 = np.array(point1)
        p2 = np.array(point2)  # Vertex point
        p3 = np.array(point3)
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def get_key_points(self, landmarks):
        """Extract key body points for squat analysis"""
        if not landmarks:
            return None
            
        points = {}
        # Key landmarks for squat analysis
        landmark_indices = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        for name, idx in landmark_indices.items():
            if idx < len(landmarks.landmark):
                landmark = landmarks.landmark[idx]
                points[name] = [landmark.x, landmark.y, landmark.visibility]
        
        return points
    
    def analyze_squat_form(self, points, frame_height, frame_width):
        """Analyze squat form for front-facing videos"""
        if not points:
            return None
        
        # Calculate average positions for stability
        avg_hip_x = (points['left_hip'][0] + points['right_hip'][0]) / 2 * frame_width
        avg_hip_y = (points['left_hip'][1] + points['right_hip'][1]) / 2 * frame_height
        avg_knee_x = (points['left_knee'][0] + points['right_knee'][0]) / 2 * frame_width
        avg_knee_y = (points['left_knee'][1] + points['right_knee'][1]) / 2 * frame_height
        avg_ankle_x = (points['left_ankle'][0] + points['right_ankle'][0]) / 2 * frame_width
        
        # Calibrate standing position (first 30 frames when person is standing)
        if self.calibration_frames < 30:
            self.calibration_data.append({
                'hip_y': avg_hip_y,
                'knee_y': avg_knee_y
            })
            self.calibration_frames += 1
            
            if self.calibration_frames == 30:
                # Set standing reference points
                self.standing_hip_height = np.mean([d['hip_y'] for d in self.calibration_data])
                self.standing_knee_height = np.mean([d['knee_y'] for d in self.calibration_data])
                print(f"Calibration complete! Standing hip: {self.standing_hip_height:.1f}, knee: {self.standing_knee_height:.1f}")
        
        if self.standing_hip_height is None:
            return None  # Still calibrating
        
        # 1. Depth Analysis - How much hip has dropped from standing
        hip_drop = avg_hip_y - self.standing_hip_height
        standing_height = self.standing_knee_height - self.standing_hip_height
        depth_percentage = (hip_drop / standing_height) * 100 if standing_height > 0 else 0
        
        # 2. Hip-to-knee ratio - Hip should be below knee level in good squat
        hip_knee_ratio = avg_hip_y / avg_knee_y if avg_knee_y > 0 else 0
        
        # 3. Knee Alignment - Knees should track over ankles, not cave in
        left_knee_x = points['left_knee'][0] * frame_width
        right_knee_x = points['right_knee'][0] * frame_width
        left_ankle_x = points['left_ankle'][0] * frame_width
        right_ankle_x = points['right_ankle'][0] * frame_width
        
        # Calculate knee tracking (closer to 1.0 = better alignment)
        left_knee_alignment = abs(left_knee_x - left_ankle_x) / frame_width * 100
        right_knee_alignment = abs(right_knee_x - right_ankle_x) / frame_width * 100
        avg_knee_alignment = (left_knee_alignment + right_knee_alignment) / 2
        
        # Convert to ratio where 1.0 = perfect, lower = worse alignment
        knee_alignment_score = max(0, 1.0 - (avg_knee_alignment / 10))  # 10% of frame width tolerance
        
        # 4. Symmetry - Left and right sides should be balanced
        left_hip_y = points['left_hip'][1] * frame_height
        right_hip_y = points['right_hip'][1] * frame_height
        left_knee_y = points['left_knee'][1] * frame_height
        right_knee_y = points['right_knee'][1] * frame_height
        
        # Calculate symmetry ratios
        hip_symmetry = min(left_hip_y, right_hip_y) / max(left_hip_y, right_hip_y) if max(left_hip_y, right_hip_y) > 0 else 0
        knee_symmetry = min(left_knee_y, right_knee_y) / max(left_knee_y, right_knee_y) if max(left_knee_y, right_knee_y) > 0 else 0
        symmetry_score = (hip_symmetry + knee_symmetry) / 2
        
        return {
            'depth_percentage': depth_percentage,
            'hip_knee_ratio': hip_knee_ratio,
            'knee_alignment': knee_alignment_score,
            'symmetry_score': symmetry_score,
            'hip_drop': hip_drop,
            'current_hip_y': avg_hip_y,
            'standing_hip_y': self.standing_hip_height
        }
    
    def score_squat_rep(self, metrics):
        """Score a squat rep based on front-facing form metrics"""
        if not metrics:
            return 0, "red"
        
        scores = []
        
        # 1. Depth Score (25-40% hip drop is ideal)
        depth_pct = metrics['depth_percentage']
        if 25 <= depth_pct <= 40:
            depth_score = 1.0  # Perfect depth
        elif 20 <= depth_pct < 25 or 40 < depth_pct <= 45:
            depth_score = 0.8  # Good depth
        elif 15 <= depth_pct < 20 or 45 < depth_pct <= 50:
            depth_score = 0.6  # Acceptable depth
        else:
            depth_score = 0.0  # Poor depth
        
        # 2. Hip-Knee Ratio Score (hip should be below knee)
        hip_knee_ratio = metrics['hip_knee_ratio']
        if 1.05 <= hip_knee_ratio <= 1.25:
            ratio_score = 1.0  # Good depth
        elif 1.0 <= hip_knee_ratio < 1.05:
            ratio_score = 0.6  # Shallow but acceptable
        else:
            ratio_score = 0.0  # Too shallow or too deep
        
        # 3. Knee Alignment Score (knees tracking over feet)
        knee_alignment = metrics['knee_alignment']
        if knee_alignment >= 0.9:
            alignment_score = 1.0  # Excellent alignment
        elif knee_alignment >= 0.8:
            alignment_score = 0.8  # Good alignment
        elif knee_alignment >= 0.7:
            alignment_score = 0.6  # Acceptable alignment
        else:
            alignment_score = 0.0  # Poor alignment (knee cave)
        
        # 4. Symmetry Score (left-right balance)
        symmetry = metrics['symmetry_score']
        if symmetry >= 0.95:
            symmetry_score = 1.0  # Excellent symmetry
        elif symmetry >= 0.9:
            symmetry_score = 0.8  # Good symmetry
        elif symmetry >= 0.85:
            symmetry_score = 0.6  # Acceptable symmetry
        else:
            symmetry_score = 0.0  # Poor symmetry
        
        # Calculate weighted overall score
        # Depth and hip-knee ratio are most important for front-facing analysis
        overall_score = (
            depth_score * 0.4 +        # 40% weight - depth is crucial
            ratio_score * 0.3 +        # 30% weight - proper depth relative to knees
            alignment_score * 0.2 +    # 20% weight - knee tracking
            symmetry_score * 0.1       # 10% weight - balance
        )
        
        # Determine color category
        if overall_score >= 0.85:
            return overall_score, "green"
        elif overall_score >= 0.6:
            return overall_score, "yellow"
        else:
            return overall_score, "red"
    
    def detect_squat_reps(self, points, frame_height):
        """Detect squat repetitions based on hip movement"""
        if not points:
            return
            
        # Calculate average hip height (normalized)
        avg_hip_y = (points['left_hip'][1] + points['right_hip'][1]) / 2
        self.position_history.append(avg_hip_y)
        
        if len(self.position_history) < 10:
            return
            
        current_pos = np.mean(list(self.position_history)[-5:])  # Smooth current position
        previous_pos = np.mean(list(self.position_history)[-15:-5])  # Previous position
        
        # State machine for squat detection
        if self.squat_state == "standing":
            if current_pos - previous_pos > self.descent_threshold:  # Moving down
                self.squat_state = "descending"
                self.current_rep_data = {'start_time': time.time()}
                
        elif self.squat_state == "descending":
            if current_pos - previous_pos < -self.ascent_threshold:  # Started moving up
                self.squat_state = "ascending"
                # Record bottom position metrics
                metrics = self.analyze_squat_form(points, frame_height, 640)  # Assuming width 640
                if metrics:
                    self.current_rep_data['bottom_metrics'] = metrics
                    
        elif self.squat_state == "ascending":
            if abs(current_pos - previous_pos) < 0.01:  # Back to standing
                self.squat_state = "standing"
                self.rep_count += 1
                
                # Score the completed rep
                if 'bottom_metrics' in self.current_rep_data:
                    score, color = self.score_squat_rep(self.current_rep_data['bottom_metrics'])
                    self.rep_scores.append({
                        'rep': self.rep_count,
                        'score': score,
                        'color': color,
                        'metrics': self.current_rep_data['bottom_metrics']
                    })
                
                self.current_rep_data = {}
    
    def draw_analysis(self, frame, landmarks):
        """Draw pose landmarks and analysis results on frame"""
        if landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw rep counter and scores
        y_offset = 30
        cv2.putText(frame, f"Reps: {self.rep_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show last 3 rep scores
        y_offset += 40
        for i, rep_data in enumerate(self.rep_scores[-3:]):
            color_map = {'green': (0, 255, 0), 'yellow': (0, 255, 255), 'red': (0, 0, 255)}
            color = color_map.get(rep_data['color'], (255, 255, 255))
            
            text = f"Rep {rep_data['rep']}: {rep_data['score']:.1f} pts"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        # Show total score
        total_score = sum(rep['score'] for rep in self.rep_scores)
        cv2.putText(frame, f"Total Score: {total_score:.1f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show current state
        y_offset += 40
        cv2.putText(frame, f"State: {self.squat_state}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        return frame
    
    def process_video(self, video_path, show_output=True, save_output=False, output_path=None):
        """Process video file and analyze squats"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video file '{video_path}'")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print("Squat Form Analyzer Started!")
        print(f"Video: {video_path}")
        print(f"Duration: {duration:.1f} seconds, FPS: {fps}, Frames: {total_frames}")
        print("\nPerfect Squat Criteria (Front-Facing):")
        print("- Depth: 25-40% hip drop from standing")
        print("- Hip Position: Below knee level")
        print("- Knee Alignment: Knees track over feet") 
        print("- Symmetry: Balanced left-right movement")
        print("\nScoring:")
        print("- Green: 0.85+ pts (Excellent form)")
        print("- Yellow: 0.6+ pts (Good form)")
        print("- Red: <0.6 pts (Needs improvement)")
        
        if show_output:
            print("\nPress 'q' to quit, 's' to skip to end, SPACE to pause")
        
        # Setup video writer if saving output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        if save_output and output_path:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        paused = False
        
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                points = self.get_key_points(results.pose_landmarks)
                self.detect_squat_reps(points, frame.shape[0])
            
            # Draw analysis
            annotated_frame = self.draw_analysis(frame, results.pose_landmarks)
            
            # Add progress info
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            time_elapsed = frame_count / fps if fps > 0 else 0
            cv2.putText(annotated_frame, f"Progress: {progress:.1f}% ({time_elapsed:.1f}s)", 
                       (10, annotated_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save frame if required
            if out is not None:
                out.write(annotated_frame)
            
            # Show video if requested
            if show_output:
                cv2.imshow('Squat Form Analyzer', annotated_frame)
                
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Skip to end
                    break
                elif key == ord(' '):  # Pause/unpause
                    paused = not paused
                    print("Paused" if paused else "Resumed")
            
            # Print progress every 30 frames
            if frame_count % 30 == 0:
                print(f"Processing... {progress:.1f}% complete")
        
        cap.release()
        if out is not None:
            out.release()
        if show_output:
            cv2.destroyAllWindows()
        
        # Print final results
        print(f"\n{'='*50}")
        print(f"FINAL ANALYSIS RESULTS")
        print(f"{'='*50}")
        print(f"Video: {video_path}")
        print(f"Total Reps Detected: {self.rep_count}")
        print(f"Total Score: {sum(rep['score'] for rep in self.rep_scores):.1f} / {self.rep_count}")
        
        if self.rep_scores:
            green_reps = sum(1 for rep in self.rep_scores if rep['color'] == 'green')
            yellow_reps = sum(1 for rep in self.rep_scores if rep['color'] == 'yellow')
            red_reps = sum(1 for rep in self.rep_scores if rep['color'] == 'red')
            
            print(f"\nRep Breakdown:")
            print(f"  🟢 Green Reps: {green_reps} (Perfect form)")
            print(f"  🟡 Yellow Reps: {yellow_reps} (Good form)")
            print(f"  🔴 Red Reps: {red_reps} (Poor form)")
            print(f"\nForm Quality: {((green_reps + yellow_reps*0.6) / self.rep_count * 100):.1f}%")
            
            # Detailed rep-by-rep breakdown
            print(f"\nDetailed Rep Analysis:")
            print(f"{'Rep':<4} {'Score':<6} {'Quality':<8} {'Depth%':<7} {'Hip/Knee':<8} {'Alignment':<10} {'Symmetry':<8}")
            print("-" * 60)
            
            for rep_data in self.rep_scores:
                metrics = rep_data['metrics']
                print(f"{rep_data['rep']:<4} "
                      f"{rep_data['score']:<6.2f} "
                      f"{rep_data['color']:<8} "
                      f"{metrics['depth_percentage']:<7.1f} "
                      f"{metrics['hip_knee_ratio']:<8.2f} "
                      f"{metrics['knee_alignment']:<10.2f} "
                      f"{metrics['symmetry_score']:<8.2f}")
        
        return {
            'total_reps': self.rep_count,
            'total_score': sum(rep['score'] for rep in self.rep_scores),
            'rep_details': self.rep_scores,
            'video_info': {
                'path': video_path,
                'duration': duration,
                'fps': fps,
                'total_frames': total_frames
            }
        }

def main():
    import sys
    import os
    
    # Get video file path from command line or user input
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = input("Enter video file path: ").strip().strip('"')
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return
    
    # Ask user for processing options
    print("\nProcessing Options:")
    print("1. Analyze with video display (default)")
    print("2. Analyze without display (faster)")
    print("3. Analyze and save annotated video")
    
    choice = input("Enter choice (1-3) or press Enter for default: ").strip()
    
    show_output = True
    save_output = False
    output_path = None
    
    if choice == "2":
        show_output = False
    elif choice == "3":
        show_output = True
        save_output = True
        output_path = input("Enter output video path (e.g., 'analyzed_video.mp4'): ").strip()
        if not output_path:
            output_path = "squat_analysis_output.mp4"
    
    # Create analyzer instance
    analyzer = SquatFormAnalyzer()
    
    # Process the video
    print(f"\nStarting analysis of: {video_path}")
    results = analyzer.process_video(
        video_path=video_path, 
        show_output=show_output,
        save_output=save_output,
        output_path=output_path
    )
    
    if save_output and output_path:
        print(f"\nAnnotated video saved to: {output_path}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
    
    