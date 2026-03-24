from collections import deque
from typing import Dict, Optional, List
import matplotlib.pyplot as plt
import numpy as np


class RepCounter:
    """Universal rep counter for various exercises with improved logic and form categorization"""

    def __init__(self,
                 exercise_type: str,
                 window_size: int = 120,
                 smoothing_window: int = 5):
        self.exercise_type = exercise_type
        self.window_size = window_size
        self.smoothing_window = smoothing_window

        # Exercise-specific configurations - FIXED THRESHOLDS
        self.exercise_configs = {
            'squat': {
                'primary_angle': 'knee',
                'secondary_angle': 'hip',
                'down_threshold': 100, 
                'up_threshold': 150,  
                'min_range': 30,     
                'confidence_threshold': 0.2,
                'peak_prominence': 10,
                'min_rep_duration': 15,
                'max_rep_duration': 150,
                # Form evaluation thresholds
                'perfect_bottom_angle': 90,  # Perfect squat depth
                'perfect_top_angle': 170,    # Perfect standing position
                'perfect_range': 80,         # Perfect range of motion
                'green_tolerance': 15,       # Within 15 degrees = green
                'orange_tolerance': 30,      # Within 30 degrees = orange
            },
            'shoulder_press': {
                'primary_angle': 'elbow',
                'secondary_angle': 'shoulder',
                'down_threshold': 85,
                'up_threshold': 165,
                'min_range': 55,
                'confidence_threshold': 0.5,
                'peak_prominence': 20,
                'min_rep_duration': 15,
                'max_rep_duration': 120,
                # Form evaluation thresholds
                'perfect_bottom_angle': 90,
                'perfect_top_angle': 175,
                'perfect_range': 85,
                'green_tolerance': 10,
                'orange_tolerance': 20,
            },
            'deadlift': {
                'primary_angle': 'hip',
                'secondary_angle': 'knee',
                'down_threshold': 85,
                'up_threshold': 165,
                'min_range': 50,
                'confidence_threshold': 0.5,
                'peak_prominence': 18,
                'min_rep_duration': 25,
                'max_rep_duration': 180,
                # Form evaluation thresholds
                'perfect_bottom_angle': 85,
                'perfect_top_angle': 170,
                'perfect_range': 85,
                'green_tolerance': 12,
                'orange_tolerance': 25,
            },
            'pushup': {
                'primary_angle': 'elbow',
                'secondary_angle': 'shoulder_horizontal',
                'down_threshold': 80,
                'up_threshold': 165,
                'min_range': 45,
                'confidence_threshold': 0.5,
                'peak_prominence': 15,
                'min_rep_duration': 18,
                'max_rep_duration': 100,
                # Form evaluation thresholds
                'perfect_bottom_angle': 90,
                'perfect_top_angle': 170,
                'perfect_range': 80,
                'green_tolerance': 12,
                'orange_tolerance': 25,
            }
        }

        self.reset()

    def reset(self):
        """Reset counter state"""
        self.rep_count = 0
        self.angle_history = deque(maxlen=self.window_size)
        self.smoothed_angles = deque(maxlen=self.window_size)
        self.state_history = deque(maxlen=self.window_size)
        self.confidence_history = deque(maxlen=self.window_size)
        self.current_state = 'neutral'  # 'up', 'down', 'neutral'
        self.last_rep_frame = -50
        self.rep_timestamps = []
        self.angle_traces = []
        self.potential_rep_start = None
        self.rep_phases = []
        
        # NEW: Add state tracking for better debugging
        self.state_changes = []
        self.debug_info = []
        
        # NEW: Form categorization tracking
        self.rep_categories = []  # Store category for each rep
        self.rep_points = []      # Store points earned for each rep
        self.form_analysis_data = []  # Store detailed form analysis

    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple:
        """Calculate angle between three points with confidence score"""
        # Check if all points are valid
        if (len(p1) > 2 and p1[2] < 0.2) or (len(p2) > 2 and p2[2] < 0.2) or (len(p3) > 2 and p3[2] < 0.2):
            return 0.0, 0.0  # Low confidence

        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]

        # Handle zero vectors
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0, 0.0

        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        # Calculate confidence based on keypoint confidence and vector lengths
        min_confidence = min(p1[2] if len(p1) > 2 else 1.0,
                             p2[2] if len(p2) > 2 else 1.0,
                             p3[2] if len(p3) > 2 else 1.0)

        # More lenient length penalty
        length_penalty = min(1.0, (norm1 + norm2) / 50.0)  # DECREASED denominator
        confidence = min_confidence * length_penalty

        return np.degrees(angle), confidence

    def get_exercise_angles(self, keypoints: np.ndarray) -> Dict[str, tuple]:
        """Calculate relevant angles for the current exercise with confidence scores"""
        angles = {}

        # Validate keypoints array
        if keypoints.shape[0] < 17:  # COCO format has 17 keypoints
            return angles

        try:
            if self.exercise_type == 'squat':
                # Left and right knee angles
                left_knee_angle, left_knee_conf = self.calculate_angle(
                    keypoints[11],  # left hip
                    keypoints[13],  # left knee
                    keypoints[15]  # left ankle
                )
                right_knee_angle, right_knee_conf = self.calculate_angle(
                    keypoints[12],  # right hip
                    keypoints[14],  # right knee
                    keypoints[16]  # right ankle
                )

                # Use weighted average based on confidence - IMPROVED LOGIC
                if left_knee_conf > 0 or right_knee_conf > 0:
                    if left_knee_conf > 0 and right_knee_conf > 0:
                        # Both sides available - use weighted average
                        total_conf = left_knee_conf + right_knee_conf
                        knee_angle = (left_knee_angle * left_knee_conf + right_knee_angle * right_knee_conf) / total_conf
                        angles['knee'] = (knee_angle, total_conf / 2)
                    elif left_knee_conf > 0:
                        # Only left side available
                        angles['knee'] = (left_knee_angle, left_knee_conf)
                    else:
                        # Only right side available
                        angles['knee'] = (right_knee_angle, right_knee_conf)

                # Hip angle with better calculation
                if all(keypoints[i][2] > 0.2 for i in [5, 6, 11, 12, 13, 14]):  # DECREASED confidence requirement
                    shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2
                    hip_center = (keypoints[11][:2] + keypoints[12][:2]) / 2
                    knee_center = (keypoints[13][:2] + keypoints[14][:2]) / 2

                    hip_angle, hip_conf = self.calculate_angle(
                        np.append(shoulder_center, 0.8),  # Give decent confidence to computed points
                        np.append(hip_center, 0.8),
                        np.append(knee_center, 0.8)
                    )
                    angles['hip'] = (hip_angle, hip_conf)

            # Add other exercise types here as needed...

        except (IndexError, ValueError) as e:
            # Handle any calculation errors gracefully
            pass

        return angles

    def apply_smoothing(self, values: list) -> float:
        """Apply Gaussian smoothing to angle values"""
        if len(values) < 2:  # DECREASED minimum requirement
            return values[-1] if values else 0

        # Use simpler smoothing for better responsiveness
        window_size = min(len(values), self.smoothing_window)
        if window_size < 2:
            return float(np.mean(values))

        # Simple moving average with slight weighting towards recent values
        recent_values = np.array(values[-window_size:])
        weights = np.linspace(0.5, 1.0, len(recent_values))
        weights /= weights.sum()
        
        return np.sum(recent_values * weights)

    def categorize_rep_form(self, rep_data: Dict) -> Dict:
        """Categorize a rep based on form analysis"""
        
        config = self.exercise_configs[self.exercise_type]
        print("Rep Data:", rep_data)
        # Get angles from rep data
        bottom_angle = rep_data['min_angle']
        top_angle = rep_data['max_angle']
        range_of_motion = rep_data['range_of_motion']
        
        # Calculate deviations from perfect form
        bottom_deviation = abs(bottom_angle - config['perfect_bottom_angle'])
        top_deviation = abs(top_angle - config['perfect_top_angle'])
        range_deviation = abs(range_of_motion - config['perfect_range'])
        
        # Find the maximum deviation (worst aspect)
        max_deviation = max(bottom_deviation, top_deviation, range_deviation)
        
        # Determine category based on maximum deviation
        if max_deviation <= config['green_tolerance']:
            category = 'GREEN'
            points = 1.0
            feedback = "Excellent form! Great technique."
        elif max_deviation <= config['orange_tolerance']:
            category = 'ORANGE'
            # Linear interpolation between green and orange thresholds
            points = 1.0 - ((max_deviation - config['green_tolerance']) / 
                           (config['orange_tolerance'] - config['green_tolerance']))
            points = max(0.1, points)  # Minimum 0.1 points for orange
            feedback = "Good form with minor deviations. Keep improving!"
        else:
            category = 'RED'
            points = 0.0
            feedback = "Form needs improvement. Focus on proper technique."
        
        # Determine primary issue for feedback
        if bottom_deviation == max_deviation:
            primary_issue = f"Bottom position: {bottom_angle:.1f}° (target: {config['perfect_bottom_angle']}°)"
        elif top_deviation == max_deviation:
            primary_issue = f"Top position: {top_angle:.1f}° (target: {config['perfect_top_angle']}°)"
        else:
            primary_issue = f"Range of motion: {range_of_motion:.1f}° (target: {config['perfect_range']}°)"
        
        return {
            'category': category,
            'points': points,
            'feedback': feedback,
            'primary_issue': primary_issue,
            'form_analysis': {
                'bottom_angle': bottom_angle,
                'top_angle': top_angle,
                'range_of_motion': range_of_motion,
                'bottom_deviation': bottom_deviation,
                'top_deviation': top_deviation,
                'range_deviation': range_deviation,
                'max_deviation': max_deviation,
                'perfect_bottom': config['perfect_bottom_angle'],
                'perfect_top': config['perfect_top_angle'],
                'perfect_range': config['perfect_range']
            },
            'rep_number': len(self.rep_categories) + 1
        }

    def process_frame(self, keypoints: np.ndarray, frame_num: int) -> Dict:
        """Process a single frame and update rep count - IMPROVED STATE MACHINE"""
        # Get exercise-specific angles
        angles = self.get_exercise_angles(keypoints)

        if not angles:
            return {
                'rep_count': self.rep_count,
                'current_state': self.current_state,
                'angles': {},
                'confidence': 0.0
            }

        config = self.exercise_configs[self.exercise_type]
        primary_angle_data = angles.get(config['primary_angle'], (0, 0))
        primary_angle, confidence = primary_angle_data

        # More lenient confidence check
        if confidence < config['confidence_threshold']:
            # Still store the data but with lower confidence
            self.angle_history.append(primary_angle)
            self.confidence_history.append(confidence)
            return {
                'rep_count': self.rep_count,
                'current_state': self.current_state,
                'angles': {k: v[0] for k, v in angles.items()},
                'confidence': confidence
            }

        # Store angle and confidence history
        self.angle_history.append(primary_angle)
        self.confidence_history.append(confidence)

        # Apply smoothing
        smoothed_angle = self.apply_smoothing(list(self.angle_history))
        self.smoothed_angles.append(smoothed_angle)

        # Store trace data
        self.angle_traces.append({
            'frame': frame_num,
            'angle': primary_angle,
            'smoothed_angle': smoothed_angle,
            'state': self.current_state,
            'confidence': confidence
        })

        # IMPROVED STATE MACHINE
        prev_state = self.current_state

        # State transitions - SIMPLIFIED AND MORE ROBUST
        if smoothed_angle <= config['down_threshold']:
            if self.current_state != 'down':
                print(f"Frame {frame_num}: Transitioning to DOWN (angle: {smoothed_angle:.1f}°)")
                self.current_state = 'down'
                self.state_changes.append((frame_num, 'down', smoothed_angle))
                if self.potential_rep_start is None:
                    self.potential_rep_start = frame_num
                    print(f"Frame {frame_num}: Starting potential rep")
                    
        elif smoothed_angle >= config['up_threshold']:
            if self.current_state != 'up':
                print(f"Frame {frame_num}: Transitioning to UP (angle: {smoothed_angle:.1f}°)")
                self.current_state = 'up'
                self.state_changes.append((frame_num, 'up', smoothed_angle))
                
                # Check for rep completion - SIMPLIFIED LOGIC
                if (self.potential_rep_start is not None and 
                    prev_state == 'down'):
                    
                    rep_duration = frame_num - self.potential_rep_start
                    
                    # Check duration constraints
                    if (rep_duration >= config['min_rep_duration'] and 
                        rep_duration <= config.get('max_rep_duration', 200) and
                        frame_num - self.last_rep_frame > config['min_rep_duration']):
                        
                        # Check range of motion from the actual rep period
                        if len(self.smoothed_angles) >= rep_duration:
                            rep_angles = list(self.smoothed_angles)[-rep_duration:]
                            angle_range = max(rep_angles) - min(rep_angles)
                            
                            print(f"Frame {frame_num}: Rep candidate - duration: {rep_duration}, range: {angle_range:.1f}°")
                            
                            if angle_range >= config['min_range']:
                                self.rep_count += 1
                                self.last_rep_frame = frame_num
                                self.rep_timestamps.append(frame_num)
                                
                                rep_data = {
                                    'start_frame': self.potential_rep_start,
                                    'end_frame': frame_num,
                                    'duration': rep_duration,
                                    'range_of_motion': angle_range,
                                    'min_angle': min(rep_angles),
                                    'max_angle': max(rep_angles)
                                }
                                self.rep_phases.append(rep_data)
                                
                                # NEW: Categorize the rep based on form
                                form_analysis = self.categorize_rep_form(rep_data)
                                self.rep_categories.append(form_analysis['category'])
                                self.rep_points.append(form_analysis['points'])
                                self.form_analysis_data.append(form_analysis)
                                
                                print(f"REP {self.rep_count} COMPLETED! Range: {angle_range:.1f}°, Duration: {rep_duration} frames")
                                print(f"  Category: {form_analysis['category']}, Points: {form_analysis['points']:.2f}")
                                print(f"  Issue: {form_analysis['primary_issue']}")
                            else:
                                print(f"Rep rejected - insufficient range: {angle_range:.1f}° < {config['min_range']}°")
                    else:
                        print(f"Rep rejected - duration issue: {rep_duration} frames")

                # Reset potential rep start
                self.potential_rep_start = None
        else:
            # Neutral zone
            if self.current_state not in ['up', 'down']:
                self.current_state = 'neutral'

        self.state_history.append(self.current_state)

        return {
            'rep_count': self.rep_count,
            'current_state': self.current_state,
            'angles': {k: v[0] for k, v in angles.items()},
            'primary_angle': primary_angle,
            'smoothed_angle': smoothed_angle,
            'confidence': confidence,
            'potential_rep_active': self.potential_rep_start is not None,
            'total_points': sum(self.rep_points),
            'current_categories': self.rep_categories.copy()
        }

    def get_form_summary(self) -> Dict:
        """Get comprehensive form analysis summary"""
        if not self.rep_categories:
            return {
                'total_points': 0.0,
                'average_score': 0.0,
                'percentage_score': 0.0,
                'form_distribution': {'GREEN': 0, 'ORANGE': 0, 'RED': 0},
                'form_percentages': {'GREEN': 0.0, 'ORANGE': 0.0, 'RED': 0.0},
                'recommendations': ["No reps completed yet."],
                'rep_details': []
            }
        
        # Calculate distribution
        green_count = self.rep_categories.count('GREEN')
        orange_count = self.rep_categories.count('ORANGE')
        red_count = self.rep_categories.count('RED')
        total_reps = len(self.rep_categories)
        
        # Calculate points and percentages
        total_points = sum(self.rep_points)
        average_score = total_points / total_reps if total_reps > 0 else 0
        percentage_score = (average_score * 100)
        
        # Generate recommendations
        recommendations = []
        if red_count > total_reps * 0.5:
            recommendations.append("Focus on proper form - many reps had significant deviations")
        if green_count > total_reps * 0.7:
            recommendations.append("Excellent form consistency! Keep it up!")
        elif orange_count > total_reps * 0.5:
            recommendations.append("Good form overall, small improvements will help")
        
        # Add exercise-specific recommendations
        if self.form_analysis_data:
            # Find most common issue
            bottom_issues = sum(1 for analysis in self.form_analysis_data 
                              if 'Bottom position' in analysis['primary_issue'])
            top_issues = sum(1 for analysis in self.form_analysis_data 
                           if 'Top position' in analysis['primary_issue'])
            range_issues = sum(1 for analysis in self.form_analysis_data 
                             if 'Range of motion' in analysis['primary_issue'])
            
            if bottom_issues > max(top_issues, range_issues):
                recommendations.append("Focus on achieving proper depth/bottom position")
            elif top_issues > max(bottom_issues, range_issues):
                recommendations.append("Focus on completing full extension at the top")
            elif range_issues > max(bottom_issues, top_issues):
                recommendations.append("Work on achieving full range of motion")
        
        return {
            'total_points': total_points,
            'average_score': average_score,
            'percentage_score': int(percentage_score),
            'form_distribution': {
                'GREEN': green_count,
                'ORANGE': orange_count,
                'RED': red_count
            },
            'form_percentages': {
                'GREEN': (green_count / total_reps * 100) if total_reps > 0 else 0,
                'ORANGE': (orange_count / total_reps * 100) if total_reps > 0 else 0,
                'RED': (red_count / total_reps * 100) if total_reps > 0 else 0
            },
            'recommendations': recommendations if recommendations else ["Keep practicing to improve form!"],
            'rep_details': self.form_analysis_data
        }

    def print_workout_summary(self):
        """Print a comprehensive workout summary"""
        form_summary = self.get_form_summary()
        
        print("=" * 60)
        print(f"WORKOUT SUMMARY - {self.exercise_type.upper()}")
        print("=" * 60)
        print(f"Total Reps Completed: {self.rep_count}")
        print(f"Total Points Earned: {form_summary['total_points']:.2f}")
        print(f"Average Score per Rep: {form_summary['average_score']:.2f} ({form_summary['percentage_score']:.1f}%)")
        print()
        print("FORM DISTRIBUTION:")
        print(f"🟢 GREEN (Excellent): {form_summary['form_distribution']['GREEN']} reps ({form_summary['form_percentages']['GREEN']:.1f}%)")
        print(f"🟠 ORANGE (Good): {form_summary['form_distribution']['ORANGE']} reps ({form_summary['form_percentages']['ORANGE']:.1f}%)")
        print(f"🔴 RED (Needs Work): {form_summary['form_distribution']['RED']} reps ({form_summary['form_percentages']['RED']:.1f}%)")
        print()
        print("RECOMMENDATIONS:")
        for i, rec in enumerate(form_summary['recommendations'], 1):
            print(f"{i}. {rec}")
        print()
        print("DETAILED REP ANALYSIS:")
        for i, rep_detail in enumerate(form_summary['rep_details'], 1):
            category_emoji = "🟢" if rep_detail['category'] == 'GREEN' else "🟠" if rep_detail['category'] == 'ORANGE' else "🔴"
            print(f"Rep {i}: {category_emoji} {rep_detail['category']} - {rep_detail['points']:.2f} points")
            print(f"  {rep_detail['feedback']}")
            print(f"  Issue: {rep_detail['primary_issue']}")
            print()

    def get_debug_info(self):
        """Get debugging information"""
        return {
            'state_changes': self.state_changes,
            'config': self.exercise_configs[self.exercise_type],
            'total_frames': len(self.angle_traces),
            'rep_phases': self.rep_phases,
            'form_categories': self.rep_categories,
            'form_points': self.rep_points
        }
    
    def get_rep_quality_metrics(self) -> Dict:
        """Analyze rep quality and consistency"""
        if len(self.rep_timestamps) < 1:  # Changed from 2 to 1
            return {
                'total_reps': self.rep_count,
                'avg_confidence': np.mean(list(self.confidence_history)) if self.confidence_history else 0,
                'data_quality': 'insufficient_data' if self.rep_count == 0 else 'single_rep'
            }

        # Calculate rep timing consistency
        rep_intervals = np.diff(self.rep_timestamps) if len(self.rep_timestamps) > 1 else [0]

        # Enhanced quality metrics using rep_phases
        if not self.rep_phases:
            return {'total_reps': self.rep_count, 'data_quality': 'no_complete_reps'}

        durations = [phase['duration'] for phase in self.rep_phases]
        ranges = [phase['range_of_motion'] for phase in self.rep_phases]

        # Calculate consistency scores
        if len(durations) > 1:
            duration_consistency = 1.0 - (np.std(durations) / (np.mean(durations) + 1e-6))
            range_consistency = 1.0 - (np.std(ranges) / (np.mean(ranges) + 1e-6))
        else:
            duration_consistency = 1.0
            range_consistency = 1.0

        return {
            'total_reps': self.rep_count,
            'avg_rep_duration': np.mean(rep_intervals) if len(rep_intervals) > 0 else 0,
            'avg_range_of_motion': np.mean(ranges),
            'range_consistency': max(0, range_consistency),
            'duration_consistency': max(0, duration_consistency),
            'avg_confidence': np.mean(list(self.confidence_history)) if self.confidence_history else 0,
            'rep_details': self.rep_phases,
            'quality_score': np.mean(np.array([range_consistency, duration_consistency])) * 100
        }
    
    def plot_angle_trace(self, save_path: Optional[str] = None):
        """Plot angle trace over time with rep markers and confidence"""
        if not self.angle_traces:
            return

        frames = [t['frame'] for t in self.angle_traces]
        angles = [t['angle'] for t in self.angle_traces]
        smoothed = [t['smoothed_angle'] for t in self.angle_traces]
        confidences = [t['confidence'] for t in self.angle_traces]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

        # Plot angles
        ax1.plot(frames, angles, 'b-', alpha=0.3, label='Raw Angle')
        ax1.plot(frames, smoothed, 'b-', linewidth=2, label='Smoothed Angle')

        # Mark rep completions with category colors
        for i, rep_frame in enumerate(self.rep_timestamps):
            if i < len(self.rep_categories):
                color = 'green' if self.rep_categories[i] == 'GREEN' else 'orange' if self.rep_categories[i] == 'ORANGE' else 'red'
                ax1.axvline(x=rep_frame, color=color, linestyle='--', alpha=0.8, linewidth=2, 
                           label=f'{self.rep_categories[i]} Rep' if i == 0 else "")

        # Mark rep phases
        for i, phase in enumerate(self.rep_phases):
            color = 'green' if i < len(self.rep_categories) and self.rep_categories[i] == 'GREEN' else 'orange' if i < len(self.rep_categories) and self.rep_categories[i] == 'ORANGE' else 'red'
            ax1.axvspan(phase['start_frame'], phase['end_frame'],
                        alpha=0.2, color=color, label='Rep Phase' if i == 0 else "")

        # Mark thresholds
        config = self.exercise_configs[self.exercise_type]
        ax1.axhline(y=config['down_threshold'], color='r', linestyle=':', label=f'Down threshold ({config["down_threshold"]}°)')
        ax1.axhline(y=config['up_threshold'], color='g', linestyle=':', label=f'Up threshold ({config["up_threshold"]}°)')

        ax1.set_ylabel('Angle (degrees)')
        ax1.set_title(f'{self.exercise_type.title()} - Angle Trace (Reps: {self.rep_count}, Points: {sum(self.rep_points):.1f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot confidence
        ax2.plot(frames, confidences, 'orange', linewidth=2, label='Confidence')
        ax2.axhline(y=config['confidence_threshold'], color='red', linestyle='--',
                    alpha=0.7, label=f'Min Confidence ({config["confidence_threshold"]})')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Pose Detection Confidence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # Plot state changes
        ax3.set_ylabel('State')
        ax3.set_xlabel('Frame')
        ax3.set_title('State Transitions')
        
        # Add state change markers
        for frame, state, angle in self.state_changes:
            color = 'red' if state == 'down' else 'green' if state == 'up' else 'gray'
            ax3.axvline(x=frame, color=color, alpha=0.7, label=f'{state.title()}' if frame == self.state_changes[0][0] else "")
            ax3.text(frame, 0.5, f'{state}\n{angle:.0f}°', rotation=90, ha='center', va='center', fontsize=8)
        
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()