# Python file
# src/exercises/squat.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SquatPhase:
    """Represents different phases of a squat"""
    STANDING = "standing"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"

class SquatAnalyzer:
    """Detailed squat analysis including biomechanics"""
    
    def __init__(self):
        self.phase_history = []
        self.joint_angles_history = []
        self.velocity_history = []
        self.current_phase = SquatPhase.STANDING
        
        # Biomechanical thresholds
        self.thresholds = {
            'min_depth_angle': 90,  # Knee angle for proper depth
            'parallel_hip_knee_diff': 10,  # Hip should be near knee level
            'max_forward_knee_travel': 0.1,  # Relative to foot
            'max_lateral_knee_deviation': 0.05,  # Knee valgus/varus
            'min_hip_width': 1.2,  # Times shoulder width
            'max_trunk_angle': 45,  # Forward lean limit
            'max_butt_wink_angle': 15,  # Pelvis tilt at bottom
        }
    
    def analyze_squat_biomechanics(self, keypoints: np.ndarray) -> Dict:
        """Comprehensive squat biomechanics analysis"""
        # Extract relevant keypoints (COCO format)
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # Calculate joint angles
        angles = self._calculate_joint_angles(keypoints)
        
        # Detect current phase
        phase = self._detect_squat_phase(angles)
        self.phase_history.append(phase)
        
        # Analyze based on phase
        analysis = {
            'phase': phase,
            'angles': angles,
            'depth_percentage': self._calculate_depth_percentage(angles),
            'form_checks': {}
        }
        
        # Phase-specific checks
        if phase == SquatPhase.BOTTOM:
            analysis['form_checks'].update(self._check_bottom_position(keypoints, angles))
        elif phase == SquatPhase.DESCENDING:
            analysis['form_checks'].update(self._check_descent_form(keypoints, angles))
        elif phase == SquatPhase.ASCENDING:
            analysis['form_checks'].update(self._check_ascent_form(keypoints, angles))
        
        # General checks (all phases)
        analysis['form_checks'].update(self._check_general_form(keypoints, angles))
        
        # Store history
        self.joint_angles_history.append(angles)
        
        return analysis
    
    def _calculate_joint_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Calculate all relevant joint angles"""
        angles = {}
        
        # Knee angles
        angles['left_knee'] = self._calculate_angle(
            keypoints[11][:2],  # left hip
            keypoints[13][:2],  # left knee
            keypoints[15][:2]   # left ankle
        )
        angles['right_knee'] = self._calculate_angle(
            keypoints[12][:2],  # right hip
            keypoints[14][:2],  # right knee
            keypoints[16][:2]   # right ankle
        )
        angles['avg_knee'] = (angles['left_knee'] + angles['right_knee']) / 2
        
        # Hip angles
        shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2
        hip_center = (keypoints[11][:2] + keypoints[12][:2]) / 2
        knee_center = (keypoints[13][:2] + keypoints[14][:2]) / 2
        
        angles['hip'] = self._calculate_angle(
            shoulder_center,
            hip_center,
            knee_center
        )
        
        # Ankle angles (dorsiflexion)
        angles['left_ankle'] = self._calculate_angle(
            keypoints[13][:2],  # knee
            keypoints[15][:2],  # ankle
            keypoints[15][:2] + np.array([0, 10])  # approximate foot
        )
        angles['right_ankle'] = self._calculate_angle(
            keypoints[14][:2],  # knee
            keypoints[16][:2],  # ankle  
            keypoints[16][:2] + np.array([0, 10])  # approximate foot
        )
        
        # Trunk angle (relative to vertical)
        trunk_vector = shoulder_center - hip_center
        angles['trunk'] = np.degrees(np.arctan2(trunk_vector[0], -trunk_vector[1]))
        
        return angles
    
    def _detect_squat_phase(self, angles: Dict[str, float]) -> str:
        """Detect current phase of squat based on angles and history"""
        knee_angle = angles['avg_knee']
        
        # Use simple state machine with hysteresis
        if len(self.phase_history) == 0:
            return SquatPhase.STANDING
        
        prev_phase = self.phase_history[-1]
        
        # Phase transitions
        if prev_phase == SquatPhase.STANDING:
            if knee_angle < 160:  # Started descending
                return SquatPhase.DESCENDING
        
        elif prev_phase == SquatPhase.DESCENDING:
            if knee_angle < 100:  # Reached bottom
                return SquatPhase.BOTTOM
            elif knee_angle > angles.get('prev_knee', 180):  # Started ascending
                return SquatPhase.ASCENDING
        
        elif prev_phase == SquatPhase.BOTTOM:
            if knee_angle > 110:  # Started ascending
                return SquatPhase.ASCENDING
        
        elif prev_phase == SquatPhase.ASCENDING:
            if knee_angle > 160:  # Back to standing
                return SquatPhase.STANDING
        
        return prev_phase
    
    def _calculate_depth_percentage(self, angles: Dict[str, float]) -> float:
        """Calculate squat depth as percentage (0=standing, 100=full depth)"""
        knee_angle = angles['avg_knee']
        
        # Map knee angle to depth percentage
        # 180° = 0% (standing), 90° = 100% (full depth)
        depth = (180 - knee_angle) / 90 * 100
        return np.clip(depth, 0, 100)
    
    def _check_bottom_position(self, keypoints: np.ndarray, angles: Dict[str, float]) -> Dict:
        """Check form at bottom of squat"""
        checks = {}
        
        # Check depth
        if angles['avg_knee'] > self.thresholds['min_depth_angle']:
            checks['insufficient_depth'] = {
                'passed': False,
                'value': angles['avg_knee'],
                'threshold': self.thresholds['min_depth_angle'],
                'message': 'Not reaching full depth - hips should be at or below knees'
            }
        else:
            checks['insufficient_depth'] = {'passed': True}
        
        # Check hip position relative to knees
        hip_y = (keypoints[11, 1] + keypoints[12, 1]) / 2
        knee_y = (keypoints[13, 1] + keypoints[14, 1]) / 2
        hip_knee_diff = hip_y - knee_y
        
        if abs(hip_knee_diff) > self.thresholds['parallel_hip_knee_diff']:
            checks['hip_position'] = {
                'passed': False,
                'value': hip_knee_diff,
                'message': 'Hips should be parallel with knees at bottom'
            }
        else:
            checks['hip_position'] = {'passed': True}
        
        # Check for butt wink (excessive pelvis tilt)
        # This would require more keypoints or pelvis tracking
        
        return checks
    
    def _check_descent_form(self, keypoints: np.ndarray, angles: Dict[str, float]) -> Dict:
        """Check form during descent phase"""
        checks = {}
        
        # Check descent speed (would need velocity calculation)
        # Check for knee initiation vs hip initiation
        # Check weight distribution
        
        return checks
    
    def _check_ascent_form(self, keypoints: np.ndarray, angles: Dict[str, float]) -> Dict:
        """Check form during ascent phase"""
        checks = {}
        
        # Check for hip drive
        # Check for knee lockout at top
        # Check for good morning squat pattern
        
        return checks
    
    def _check_general_form(self, keypoints: np.ndarray, angles: Dict[str, float]) -> Dict:
        """General form checks for all phases"""
        checks = {}
        
        # Check knee alignment (valgus/varus)
        left_knee = keypoints[13][:2]
        right_knee = keypoints[14][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        left_ankle = keypoints[15][:2]
        right_ankle = keypoints[16][:2]
        
        # Calculate knee deviation from hip-ankle line
        left_deviation = self._point_line_distance(left_knee, left_hip, left_ankle)
        right_deviation = self._point_line_distance(right_knee, right_hip, right_ankle)
        
        knee_width = abs(left_knee[0] - right_knee[0])
        hip_width = abs(left_hip[0] - right_hip[0])
        
        if knee_width < hip_width * 0.8:  # Knees caving in
            checks['knee_valgus'] = {
                'passed': False,
                'value': knee_width / hip_width,
                'message': 'Knees caving inward - push knees out over toes'
            }
        else:
            checks['knee_valgus'] = {'passed': True}
        
        # Check forward lean
        if abs(angles['trunk']) > self.thresholds['max_trunk_angle']:
            checks['forward_lean'] = {
                'passed': False,
                'value': angles['trunk'],
                'threshold': self.thresholds['max_trunk_angle'],
                'message': 'Excessive forward lean - keep chest up'
            }
        else:
            checks['forward_lean'] = {'passed': True}
        
        # Check stance width
        shoulder_width = abs(keypoints[5, 0] - keypoints[6, 0])
        stance_width = abs(left_ankle[0] - right_ankle[0])
        stance_ratio = stance_width / shoulder_width
        
        if stance_ratio < self.thresholds['min_hip_width']:
            checks['stance_width'] = {
                'passed': False,
                'value': stance_ratio,
                'threshold': self.thresholds['min_hip_width'],
                'message': 'Stance too narrow - widen feet to shoulder width or wider'
            }
        else:
            checks['stance_width'] = {'passed': True}
        
        return checks
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def _point_line_distance(self, point: np.ndarray, line_p1: np.ndarray, line_p2: np.ndarray) -> float:
        """Calculate perpendicular distance from point to line"""
        line_vec = line_p2 - line_p1
        point_vec = point - line_p1
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return float(np.linalg.norm(point_vec))
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        proj = line_p1 + proj_length * line_unitvec
        
        return float(np.linalg.norm(point - proj))
    
    def get_squat_summary(self) -> Dict:
        """Get comprehensive squat analysis summary"""
        if not self.joint_angles_history:
            return {}
        
        # Calculate statistics across all frames
        all_knee_angles = [a['avg_knee'] for a in self.joint_angles_history]
        all_hip_angles = [a['hip'] for a in self.joint_angles_history]
        all_trunk_angles = [a['trunk'] for a in self.joint_angles_history]
        
        # Count phases
        phase_counts = {}
        for phase in self.phase_history:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        return {
            'total_frames': len(self.joint_angles_history),
            'phase_distribution': phase_counts,
            'angle_ranges': {
                'knee': {
                    'min': np.min(all_knee_angles),
                    'max': np.max(all_knee_angles),
                    'range': np.max(all_knee_angles) - np.min(all_knee_angles)
                },
                'hip': {
                    'min': np.min(all_hip_angles),
                    'max': np.max(all_hip_angles),
                    'range': np.max(all_hip_angles) - np.min(all_hip_angles)
                },
                'trunk': {
                    'min': np.min(all_trunk_angles),
                    'max': np.max(all_trunk_angles),
                    'avg': np.mean(all_trunk_angles)
                }
            },
            'max_depth_achieved': 100 - (np.min(all_knee_angles) / 180 * 100),
            'consistency_score': 1.0 - np.std(all_knee_angles) / np.mean(all_knee_angles)
        }