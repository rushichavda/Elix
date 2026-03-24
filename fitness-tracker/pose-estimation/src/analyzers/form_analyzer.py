# Python file
# src/analyzers/form_analyzer.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import math


class FaultSeverity(Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class FormFault:
    fault_type: str
    severity: FaultSeverity
    description: str
    frame: int
    confidence: float
    correction: str


class FormAnalyzer:
    """Analyzes exercise form and detects common faults"""

    def __init__(self, exercise_type: str):
        self.exercise_type = exercise_type
        self.fault_history = []
        self.frame_count = 0
        self.keypoint_history = deque(maxlen=30)  # Store recent keypoints for temporal analysis
        self.fault_counts = defaultdict(int)
        self.baseline_measurements = {}  # Store baseline body measurements
        self.temporal_window = 10  # Frames to consider for temporal consistency
        
        # Fault suppression and debouncing
        self.active_faults = {}  # Track currently active faults
        self.fault_cooldown = {}  # Prevent re-triggering
        self.cooldown_frames = 30  # Frames before same fault can trigger again
        self.fault_start_frame = {}  # Track when each fault started
        
        # Confidence thresholds
        self.min_keypoint_confidence = 0.3
        self.min_fault_confidence = 0.7  # Don't report faults below this confidence
        
        # Frame checking interval (to reduce excessive checking)
        self.check_interval = 3  # Check every 3rd frame

        # Enhanced exercise-specific fault detection rules
        self.fault_rules = {
            'squat': {
                'knee_valgus': self._check_knee_valgus,
                'forward_lean': self._check_forward_lean,
                'heel_raise': self._check_heel_raise,
                'depth': self._check_squat_depth,
                'knee_forward': self._check_knee_forward,
                'asymmetry': self._check_squat_asymmetry,
                'knee_cave': self._check_knee_cave,
            },
            'shoulder_press': {
                'elbow_flare': self._check_elbow_flare,
                'wrist_bend': self._check_wrist_bend,
                'back_arch': self._check_back_arch,
                'asymmetry': self._check_asymmetry,
                'partial_rom': self._check_shoulder_press_rom,
                'forward_head': self._check_forward_head_posture,
            },
            'deadlift': {
                'rounded_back': self._check_rounded_back,
                'bar_path': self._check_bar_path,
                'hip_hinge': self._check_hip_hinge,
                'knee_lockout': self._check_knee_lockout,
                'uneven_hips': self._check_uneven_hips,
                'heel_lift': self._check_deadlift_heel_lift,
            },
            'pushup': {
                'elbow_flare': self._check_pushup_elbow_flare,
                'hip_sag': self._check_hip_sag,
                'incomplete_rom': self._check_pushup_rom,
                'neck_position': self._check_neck_position,
                'hand_position': self._check_hand_position,
                'body_alignment': self._check_pushup_body_alignment,
            }
        }

    def analyze_form(self, keypoints: np.ndarray, rep_state: str = 'neutral') -> List[FormFault]:
        """Analyze form for the current frame with fault suppression"""
        self.frame_count += 1
        
        # Only check every Nth frame to reduce computation
        if self.frame_count % self.check_interval != 0:
            return []
        
        detected_faults = []

        # Validate keypoints
        if not self._validate_keypoints(keypoints):
            return detected_faults

        # Store keypoints for temporal analysis
        self.keypoint_history.append({
            'frame': self.frame_count,
            'keypoints': keypoints.copy(),
            'rep_state': rep_state
        })

        # Initialize baseline measurements if needed
        if not self.baseline_measurements and len(self.keypoint_history) > 5:
            self._establish_baseline_measurements()

        # Get relevant fault checks for the exercise
        if self.exercise_type not in self.fault_rules:
            return detected_faults

        # Track which faults are detected this frame
        current_frame_faults = set()

        # Run each fault detection rule
        for fault_name, check_func in self.fault_rules[self.exercise_type].items():
            # Skip if in cooldown
            if fault_name in self.fault_cooldown:
                if self.frame_count < self.fault_cooldown[fault_name]:
                    continue
                else:
                    del self.fault_cooldown[fault_name]
            
            fault = check_func(keypoints, rep_state)
            
            # Apply confidence filtering
            if fault and fault.confidence >= self.min_fault_confidence:
                current_frame_faults.add(fault_name)
                
                # Only add fault if it's new or re-emerging after cooldown
                if fault_name not in self.active_faults:
                    fault.frame = self.frame_count
                    detected_faults.append(fault)
                    self.fault_history.append(fault)
                    self.fault_counts[fault_name] += 1
                    self.active_faults[fault_name] = self.frame_count
                    self.fault_start_frame[fault_name] = self.frame_count
        
        # Clear faults that are no longer detected
        for fault_name in list(self.active_faults.keys()):
            if fault_name not in current_frame_faults:
                # Require fault to be absent for multiple frames before clearing
                frames_since_start = self.frame_count - self.fault_start_frame.get(fault_name, self.frame_count)
                if frames_since_start > 10:  # Only clear if fault persisted for meaningful duration
                    del self.active_faults[fault_name]
                    # Start cooldown to prevent immediate re-triggering
                    self.fault_cooldown[fault_name] = self.frame_count + self.cooldown_frames

        return detected_faults

    def _validate_keypoints(self, keypoints: np.ndarray) -> bool:
        """Validate keypoint data quality"""
        if keypoints.shape[0] < 17:  # COCO format
            return False

        # Check if critical keypoints have sufficient confidence
        critical_points = [5, 6, 11, 12, 13, 14]  # shoulders, hips, knees
        valid_points = 0
        for idx in critical_points:
            if len(keypoints[idx]) > 2 and keypoints[idx][2] > self.min_keypoint_confidence:
                valid_points += 1
        
        # Require at least 4 out of 6 critical points to be valid
        return valid_points >= 4

    def _establish_baseline_measurements(self):
        """Establish baseline body measurements for relative calculations"""
        if len(self.keypoint_history) < 5:
            return

        # Calculate average body segment lengths
        shoulder_widths = []
        hip_widths = []
        torso_lengths = []

        for frame_data in list(self.keypoint_history)[-5:]:
            kp = frame_data['keypoints']

            # Shoulder width
            if kp[5][2] > 0.3 and kp[6][2] > 0.3:
                shoulder_width = np.linalg.norm(kp[5][:2] - kp[6][:2])
                shoulder_widths.append(shoulder_width)

            # Hip width
            if kp[11][2] > 0.3 and kp[12][2] > 0.3:
                hip_width = np.linalg.norm(kp[11][:2] - kp[12][:2])
                hip_widths.append(hip_width)

            # Torso length
            if all(kp[i][2] > 0.3 for i in [5, 6, 11, 12]):
                shoulder_center = (kp[5][:2] + kp[6][:2]) / 2
                hip_center = (kp[11][:2] + kp[12][:2]) / 2
                torso_length = np.linalg.norm(shoulder_center - hip_center)
                torso_lengths.append(torso_length)

        self.baseline_measurements = {
            'shoulder_width': np.mean(shoulder_widths) if shoulder_widths else 100,
            'hip_width': np.mean(hip_widths) if hip_widths else 80,
            'torso_length': np.mean(torso_lengths) if torso_lengths else 120,
        }

    def _calculate_angle_3d(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate 3D angle between three points"""
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]

        # Handle zero vectors
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        cosine_angle = np.dot(v1, v2) / (norm1 * norm2)
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _get_temporal_consistency(self, measurement_func, frames_back=5) -> tuple:
        """Get temporal consistency of a measurement"""
        if len(self.keypoint_history) < frames_back:
            return None, 0.0

        recent_frames = list(self.keypoint_history)[-frames_back:]
        measurements = []

        for frame_data in recent_frames:
            try:
                result = measurement_func(frame_data['keypoints'])
                if result is not None:
                    measurements.append(result)
            except:
                continue

        if len(measurements) < 2:
            return None, 0.0

        mean_val = np.mean(measurements)
        consistency = 1.0 - (np.std(measurements) / (mean_val + 1e-6))
        return mean_val, max(0.0, consistency)

    def _is_position_stable(self) -> bool:
        """Check if current position is stable (not mid-movement)"""
        if len(self.keypoint_history) < 3:
            return False
        
        # Check movement between recent frames
        recent_frames = list(self.keypoint_history)[-3:]
        movements = []
        
        for i in range(1, len(recent_frames)):
            prev_kp = recent_frames[i-1]['keypoints']
            curr_kp = recent_frames[i]['keypoints']
            
            # Calculate movement for key points
            key_points = [5, 6, 11, 12, 13, 14]  # shoulders, hips, knees
            total_movement = 0
            valid_points = 0
            
            for idx in key_points:
                if prev_kp[idx][2] > 0.3 and curr_kp[idx][2] > 0.3:
                    movement = np.linalg.norm(prev_kp[idx][:2] - curr_kp[idx][:2])
                    total_movement += movement
                    valid_points += 1
            
            if valid_points > 0:
                movements.append(total_movement / valid_points)
        
        # Position is stable if average movement is low
        return bool(len(movements) > 0 and np.mean(movements) < 5)

    # Enhanced Squat fault detection methods with adjusted thresholds
    def _check_knee_valgus(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for knees caving inward with temporal consistency"""
        if rep_state != 'down':
            return None

        def measure_knee_alignment(kp):
            if not all(kp[i][2] > 0.3 for i in [11, 12, 13, 14, 15, 16]):
                return None

            # Vector from ankle to hip for each leg
            left_leg_vector = kp[11][:2] - kp[15][:2]
            right_leg_vector = kp[12][:2] - kp[16][:2]

            # Knee position relative to ankle-hip line
            left_deviation = self._point_line_distance(kp[13][:2], kp[15][:2], kp[11][:2])
            right_deviation = self._point_line_distance(kp[14][:2], kp[16][:2], kp[12][:2])

            # Normalize by hip width
            hip_width = self.baseline_measurements.get('hip_width', 80)
            return min(left_deviation, right_deviation) / hip_width

        deviation, consistency = self._get_temporal_consistency(measure_knee_alignment)

        # Adjusted threshold from -0.15 to -0.20
        if deviation is not None and deviation < -0.20 and consistency > 0.5:
            severity = FaultSeverity.SEVERE if deviation < -0.35 else FaultSeverity.MODERATE
            return FormFault(
                fault_type="knee_valgus",
                severity=severity,
                description=f"Knees caving inward (deviation: {deviation:.2f})",
                frame=0,
                confidence=consistency,
                correction="Push your knees out in line with your toes and strengthen your glutes"
            )
        return None

    def _check_forward_lean(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for excessive forward lean with improved calculation"""

        def measure_trunk_angle(kp):
            if not all(kp[i][2] > 0.3 for i in [5, 6, 11, 12]):
                return None

            shoulder_center = (kp[5][:2] + kp[6][:2]) / 2
            hip_center = (kp[11][:2] + kp[12][:2]) / 2

            # Calculate angle from vertical
            trunk_vector = shoulder_center - hip_center
            vertical_vector = np.array([0, -1])  # Pointing up

            dot_product = np.dot(trunk_vector, vertical_vector)
            norms = np.linalg.norm(trunk_vector) * np.linalg.norm(vertical_vector)

            if norms < 1e-6:
                return None

            angle = np.degrees(np.arccos(np.clip(dot_product / norms, -1.0, 1.0)))
            return angle

        angle, consistency = self._get_temporal_consistency(measure_trunk_angle)

        # Adjusted threshold from 25 to 35 degrees
        if angle is not None and angle > 35 and consistency > 0.6:
            severity = FaultSeverity.SEVERE if angle > 50 else FaultSeverity.MODERATE
            return FormFault(
                fault_type="forward_lean",
                severity=severity,
                description=f"Excessive forward lean ({angle:.1f}° from vertical)",
                frame=0,
                confidence=consistency,
                correction="Keep your chest up, engage your core, and sit back more"
            )
        return None

    def _check_heel_raise(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check if heels are coming off the ground"""
        if rep_state != 'down':
            return None

        def measure_ankle_stability(kp):
            if not all(kp[i][2] > 0.3 for i in [15, 16]):
                return None

            # Compare current ankle height to baseline (if available)
            if len(self.keypoint_history) < 10:
                return None

            # Get baseline ankle positions from standing position
            standing_frames = [f for f in self.keypoint_history
                               if f['rep_state'] == 'up' or f['rep_state'] == 'neutral']

            if len(standing_frames) < 3:
                return None

            baseline_left_y = np.mean([f['keypoints'][15][1] for f in standing_frames[-3:]])
            baseline_right_y = np.mean([f['keypoints'][16][1] for f in standing_frames[-3:]])

            current_left_y = kp[15][1]
            current_right_y = kp[16][1]

            # Check for heel raise (negative because y increases downward)
            left_raise = baseline_left_y - current_left_y
            right_raise = baseline_right_y - current_right_y

            return max(left_raise, right_raise)

        heel_raise, consistency = self._get_temporal_consistency(measure_ankle_stability)

        # Adjusted threshold from 15 to 30 pixels
        if heel_raise is not None and heel_raise > 30 and consistency > 0.7:
            return FormFault(
                fault_type="heel_raise",
                severity=FaultSeverity.MODERATE,
                description="Heels coming off the ground",
                frame=0,
                confidence=consistency,
                correction="Keep your heels planted, work on ankle mobility"
            )
        return None

    def _check_squat_depth(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check if squat reaches proper depth"""
        if rep_state != 'down':
            return None
        
        # Only check when position is stable
        if not self._is_position_stable():
            return None

        if not all(keypoints[i][2] > 0.3 for i in [11, 12, 13, 14]):
            return None

        # Check if hips are below knees
        hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
        knee_y = (keypoints[13][1] + keypoints[14][1]) / 2

        depth_ratio = (hip_y - knee_y) / self.baseline_measurements.get('torso_length', 120)

        # Adjusted threshold from 0.1 to -0.05 (more lenient)
        if depth_ratio < -0.05:  # Hip should be noticeably below knee
            return FormFault(
                fault_type="insufficient_depth",
                severity=FaultSeverity.MINOR,
                description="Not reaching full squat depth",
                frame=0,
                confidence=0.8,
                correction="Lower your hips below your knees, work on mobility if needed"
            )
        return None

    def _check_knee_forward(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check if knees travel too far forward"""
        if rep_state != 'down':
            return None

        if not all(keypoints[i][2] > 0.3 for i in [13, 14, 15, 16]):
            return None

        # Check knee position relative to ankle
        left_knee_forward = keypoints[13][0] - keypoints[15][0]
        right_knee_forward = keypoints[14][0] - keypoints[16][0]

        # Normalize by foot length (estimated)
        foot_length = self.baseline_measurements.get('hip_width', 80) * 0.3
        max_forward = max(abs(left_knee_forward), abs(right_knee_forward)) / foot_length

        # Adjusted threshold from 1.5 to 2.0 foot lengths
        if max_forward > 2.0:  # More lenient
            return FormFault(
                fault_type="knee_forward",
                severity=FaultSeverity.MINOR,
                description="Knees traveling too far forward",
                frame=0,
                confidence=0.7,
                correction="Sit back more and initiate with your hips"
            )
        return None

    def _check_squat_asymmetry(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for left-right asymmetry in squat"""
        if not all(keypoints[i][2] > 0.3 for i in [11, 12, 13, 14]):
            return None

        # Check hip and knee asymmetry
        hip_asymmetry = abs(keypoints[11][1] - keypoints[12][1])
        knee_asymmetry = abs(keypoints[13][1] - keypoints[14][1])

        torso_length = self.baseline_measurements.get('torso_length', 120)
        relative_asymmetry = max(hip_asymmetry, knee_asymmetry) / torso_length

        # Adjusted threshold from 0.1 to 0.15
        if relative_asymmetry > 0.15:
            return FormFault(
                fault_type="asymmetry",
                severity=FaultSeverity.MODERATE,
                description=f"Left-right imbalance detected",
                frame=0,
                confidence=0.8,
                correction="Focus on balanced movement and address any mobility restrictions"
            )
        return None

    def _check_knee_cave(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Enhanced knee cave detection using multiple angles"""
        if rep_state != 'down':
            return None

        if not all(keypoints[i][2] > 0.3 for i in [11, 12, 13, 14, 15, 16]):
            return None

        # Calculate Q-angle approximation
        left_q_angle = self._calculate_angle_3d(keypoints[11], keypoints[13], keypoints[15])
        right_q_angle = self._calculate_angle_3d(keypoints[12], keypoints[14], keypoints[16])

        # Adjusted threshold from 160 to 150 degrees
        if min(left_q_angle, right_q_angle) < 150:
            severity = FaultSeverity.SEVERE if min(left_q_angle, right_q_angle) < 130 else FaultSeverity.MODERATE
            return FormFault(
                fault_type="knee_cave",
                severity=severity,
                description=f"Knee tracking issue (Q-angle: {min(left_q_angle, right_q_angle):.1f}°)",
                frame=0,
                confidence=0.8,
                correction="Focus on external rotation and glute activation"
            )
        return None

    # Enhanced Shoulder Press fault detection
    def _check_elbow_flare(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for elbows flaring out in shoulder press"""
        if not all(keypoints[i][2] > 0.3 for i in [5, 6, 7, 8]):
            return None

        # Calculate elbow angle relative to torso
        shoulder_width = np.linalg.norm(keypoints[5][:2] - keypoints[6][:2])
        left_elbow_distance = self._point_line_distance(keypoints[7][:2], keypoints[5][:2], keypoints[6][:2])
        right_elbow_distance = self._point_line_distance(keypoints[8][:2], keypoints[5][:2], keypoints[6][:2])

        max_flare = max(left_elbow_distance, right_elbow_distance) / shoulder_width

        # Adjusted threshold from 0.8 to 1.0
        if max_flare > 1.0:
            return FormFault(
                fault_type="elbow_flare",
                severity=FaultSeverity.MODERATE,
                description="Elbows flaring out too wide",
                frame=0,
                confidence=0.8,
                correction="Keep elbows at 45-degree angle to torso, not directly out to sides"
            )
        return None

    def _check_wrist_bend(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for bent wrists using forearm alignment"""
        if not all(keypoints[i][2] > 0.3 for i in [7, 8, 9, 10]):
            return None

        # Calculate forearm-wrist alignment
        left_forearm_angle = self._calculate_angle_3d(keypoints[5], keypoints[7], keypoints[9])
        right_forearm_angle = self._calculate_angle_3d(keypoints[6], keypoints[8], keypoints[10])

        min_angle = min(left_forearm_angle, right_forearm_angle)

        # Adjusted threshold from 160 to 150 degrees
        if min_angle < 150:  # Significant wrist bend
            return FormFault(
                fault_type="wrist_bend",
                severity=FaultSeverity.MINOR,
                description="Wrists bending under load",
                frame=0,
                confidence=0.7,
                correction="Keep wrists straight and in line with forearms"
            )
        return None

    def _check_back_arch(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Enhanced back arch detection"""
        if not all(keypoints[i][2] > 0.3 for i in [1, 5, 6, 11, 12]):
            return None

        # Calculate spine curvature
        head = keypoints[1][:2] if keypoints[1][2] > 0.5 else (keypoints[5][:2] + keypoints[6][:2]) / 2
        shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2
        hip_center = (keypoints[11][:2] + keypoints[12][:2]) / 2

        # Check if shoulder deviates significantly from head-hip line
        deviation = self._point_line_distance(shoulder_center, head, hip_center)
        torso_length = self.baseline_measurements.get('torso_length', 120)
        relative_deviation = deviation / torso_length

        # Adjusted threshold from 0.2 to 0.25
        if relative_deviation > 0.25:
            severity = FaultSeverity.SEVERE if relative_deviation > 0.4 else FaultSeverity.MODERATE
            return FormFault(
                fault_type="back_arch",
                severity=severity,
                description=f"Excessive back arch (deviation: {relative_deviation:.2f})",
                frame=0,
                confidence=0.9,
                correction="Engage your core and maintain neutral spine position"
            )
        return None

    def _check_asymmetry(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Enhanced asymmetry detection"""
        if not all(keypoints[i][2] > 0.3 for i in [9, 10]):
            return None

        # Compare wrist heights with temporal consistency
        def measure_wrist_asymmetry(kp):
            return abs(kp[9][1] - kp[10][1])

        asymmetry, consistency = self._get_temporal_consistency(measure_wrist_asymmetry)

        # Adjusted threshold from 20 to 30 pixels
        if asymmetry is not None and asymmetry > 30 and consistency > 0.6:
            return FormFault(
                fault_type="asymmetry",
                severity=FaultSeverity.MODERATE,
                description=f"Uneven arm position ({asymmetry:.0f}px difference)",
                frame=0,
                confidence=consistency,
                correction="Focus on moving both arms at the same pace and height"
            )
        return None

    def _check_shoulder_press_rom(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for partial range of motion in shoulder press"""
        if rep_state != 'up':
            return None
        
        # Only check when position is stable
        if not self._is_position_stable():
            return None

        if not all(keypoints[i][2] > 0.3 for i in [5, 6, 9, 10]):
            return None

        # Check if wrists are above shoulders at top position
        shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
        wrist_y = (keypoints[9][1] + keypoints[10][1]) / 2

        # Adjusted threshold from -10 to -20 pixels
        if wrist_y > shoulder_y - 20:  # Wrists should be well above shoulders
            return FormFault(
                fault_type="partial_rom",
                severity=FaultSeverity.MINOR,
                description="Not reaching full overhead position",
                frame=0,
                confidence=0.7,
                correction="Press all the way up until arms are fully extended overhead"
            )
        return None

    def _check_forward_head_posture(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for forward head posture"""
        if not all(keypoints[i][2] > 0.3 for i in [0, 1, 5, 6]):
            return None

        # Calculate head position relative to shoulders
        head_center = (keypoints[0][:2] + keypoints[1][:2]) / 2 if keypoints[0][2] > 0.3 else keypoints[1][:2]
        shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2

        # Check horizontal displacement
        forward_displacement = head_center[0] - shoulder_center[0]
        torso_length = self.baseline_measurements.get('torso_length', 120)
        relative_displacement = abs(forward_displacement) / torso_length

        # Adjusted threshold from 0.15 to 0.2
        if relative_displacement > 0.2:
            return FormFault(
                fault_type="forward_head",
                severity=FaultSeverity.MINOR,
                description="Forward head posture",
                frame=0,
                confidence=0.7,
                correction="Keep your head in neutral position over your shoulders"
            )
        return None

    # Deadlift fault detection methods (with adjusted thresholds)
    def _check_rounded_back(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for rounded back in deadlift"""
        if not all(keypoints[i][2] > 0.3 for i in [1, 5, 6, 11, 12]):
            return None

        # Calculate spine curvature using multiple points
        head = keypoints[1][:2]
        shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2
        hip_center = (keypoints[11][:2] + keypoints[12][:2]) / 2

        # Check if spine is curved (shoulder behind head-hip line)
        spine_curve = self._point_line_distance(shoulder_center, head, hip_center)
        torso_length = self.baseline_measurements.get('torso_length', 120)
        relative_curve = spine_curve / torso_length

        # Adjusted threshold from 0.15 to 0.2
        if relative_curve > 0.2:  # Significant rounding
            severity = FaultSeverity.SEVERE if relative_curve > 0.3 else FaultSeverity.MODERATE
            return FormFault(
                fault_type="rounded_back",
                severity=severity,
                description=f"Rounded back detected (curve: {relative_curve:.2f})",
                frame=0,
                confidence=0.9,
                correction="Keep chest up and maintain neutral spine throughout the lift"
            )
        return None

    def _check_bar_path(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check bar path deviation (using wrist as proxy for bar)"""
        if not all(keypoints[i][2] > 0.3 for i in [9, 10]):
            return None

        def measure_bar_path_deviation(kp):
            # Use average wrist position as bar proxy
            bar_x = (kp[9][0] + kp[10][0]) / 2

            # Compare to foot position (should be roughly vertical)
            if kp[15][2] > 0.3 and kp[16][2] > 0.3:
                foot_x = (kp[15][0] + kp[16][0]) / 2
                return abs(bar_x - foot_x)
            return None

        deviation, consistency = self._get_temporal_consistency(measure_bar_path_deviation)

        # Adjusted threshold from 30 to 40 pixels
        if deviation is not None and deviation > 40 and consistency > 0.6:
            return FormFault(
                fault_type="bar_path",
                severity=FaultSeverity.MODERATE,
                description="Bar drifting away from body",
                frame=0,
                confidence=consistency,
                correction="Keep the bar close to your body throughout the lift"
            )
        return None

    def _check_hip_hinge(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check proper hip hinge movement pattern"""
        if not all(keypoints[i][2] > 0.3 for i in [5, 6, 11, 12, 13, 14]):
            return None

        # Calculate hip angle
        shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2
        hip_center = (keypoints[11][:2] + keypoints[12][:2]) / 2
        knee_center = (keypoints[13][:2] + keypoints[14][:2]) / 2

        hip_angle = self._calculate_angle_3d(
            np.append(shoulder_center, 1.0),
            np.append(hip_center, 1.0),
            np.append(knee_center, 1.0)
        )

        # Check if hip angle is appropriate for deadlift
        if rep_state == 'down' and hip_angle > 150:  # Adjusted from 140
            return FormFault(
                fault_type="insufficient_hip_hinge",
                severity=FaultSeverity.MODERATE,
                description="Insufficient hip hinge, too much knee bend",
                frame=0,
                confidence=0.8,
                correction="Focus on sitting back with your hips, not squatting down"
            )
        return None

    def _check_knee_lockout(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for premature knee lockout"""
        if rep_state != 'up':
            return None

        if not all(keypoints[i][2] > 0.3 for i in [11, 12, 13, 14, 15, 16]):
            return None

        # Calculate knee angles
        left_knee_angle = self._calculate_angle_3d(keypoints[11], keypoints[13], keypoints[15])
        right_knee_angle = self._calculate_angle_3d(keypoints[12], keypoints[14], keypoints[16])

        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        # Check hip position - if knees are locked but hips not extended
        hip_center = (keypoints[11][:2] + keypoints[12][:2]) / 2
        shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2

        # Adjusted to be more lenient
        if avg_knee_angle > 175 and hip_center[0] > shoulder_center[0] + 10:  # Knees locked but hips back
            return FormFault(
                fault_type="knee_lockout",
                severity=FaultSeverity.MODERATE,
                description="Premature knee lockout before hip extension",
                frame=0,
                confidence=0.8,
                correction="Extend hips and knees simultaneously, don't lock knees early"
            )
        return None

    def _check_uneven_hips(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for uneven hip position"""
        if not all(keypoints[i][2] > 0.3 for i in [11, 12]):
            return None

        hip_asymmetry = abs(keypoints[11][1] - keypoints[12][1])
        hip_width = self.baseline_measurements.get('hip_width', 80)
        relative_asymmetry = hip_asymmetry / hip_width

        # Adjusted threshold from 0.15 to 0.2
        if relative_asymmetry > 0.2:
            return FormFault(
                fault_type="uneven_hips",
                severity=FaultSeverity.MODERATE,
                description="Uneven hip position",
                frame=0,
                confidence=0.8,
                correction="Focus on keeping hips level and balanced"
            )
        return None

    def _check_deadlift_heel_lift(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for heel lifting during deadlift"""
        return self._check_heel_raise(keypoints, rep_state)  # Reuse squat heel raise logic

    # Pushup fault detection methods (with adjusted thresholds)
    def _check_pushup_elbow_flare(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for excessive elbow flare in pushups"""
        if not all(keypoints[i][2] > 0.3 for i in [5, 6, 7, 8]):
            return None

        # Calculate elbow angle relative to torso
        shoulder_width = np.linalg.norm(keypoints[5][:2] - keypoints[6][:2])

        # Check elbow position relative to shoulder line
        left_elbow_flare = self._point_line_distance(keypoints[7][:2], keypoints[5][:2], keypoints[6][:2])
        right_elbow_flare = self._point_line_distance(keypoints[8][:2], keypoints[5][:2], keypoints[6][:2])

        max_flare = max(left_elbow_flare, right_elbow_flare) / shoulder_width

        # Adjusted threshold from 1.0 to 1.2
        if max_flare > 1.2:  # Elbows too far from body
            return FormFault(
                fault_type="elbow_flare",
                severity=FaultSeverity.MODERATE,
                description="Elbows flaring out too wide",
                frame=0,
                confidence=0.8,
                correction="Keep elbows closer to your body, aim for 45-degree angle"
            )
        return None

    def _check_hip_sag(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for hip sagging in pushups"""
        if not all(keypoints[i][2] > 0.3 for i in [5, 6, 11, 12, 13, 14]):
            return None

        # Calculate body line from shoulders to ankles
        shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2
        hip_center = (keypoints[11][:2] + keypoints[12][:2]) / 2
        knee_center = (keypoints[13][:2] + keypoints[14][:2]) / 2

        # Check if hips sag below straight line
        hip_deviation = self._point_line_distance(hip_center, shoulder_center, knee_center)
        torso_length = self.baseline_measurements.get('torso_length', 120)
        relative_sag = hip_deviation / torso_length

        # Adjusted threshold from 0.1 to 0.15
        if relative_sag > 0.15:
            return FormFault(
                fault_type="hip_sag",
                severity=FaultSeverity.MODERATE,
                description="Hips sagging below body line",
                frame=0,
                confidence=0.9,
                correction="Engage your core and maintain straight body line"
            )
        return None

    def _check_pushup_rom(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for incomplete range of motion in pushups"""
        if rep_state != 'down':
            return None
        
        # Only check when position is stable
        if not self._is_position_stable():
            return None

        if not all(keypoints[i][2] > 0.3 for i in [5, 6, 7, 8]):
            return None

        # Calculate elbow angles
        left_elbow_angle = self._calculate_angle_3d(keypoints[5], keypoints[7], keypoints[9])
        right_elbow_angle = self._calculate_angle_3d(keypoints[6], keypoints[8], keypoints[10])

        min_elbow_angle = min(left_elbow_angle, right_elbow_angle)

        # Adjusted threshold from 100 to 110 degrees
        if min_elbow_angle > 110:  # Not deep enough
            return FormFault(
                fault_type="incomplete_rom",
                severity=FaultSeverity.MINOR,
                description="Not lowering deep enough",
                frame=0,
                confidence=0.8,
                correction="Lower your chest closer to the ground"
            )
        return None

    def _check_neck_position(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check for neck position in pushups"""
        if not all(keypoints[i][2] > 0.3 for i in [0, 1, 5, 6]):
            return None

        # Calculate head position relative to shoulders
        head = keypoints[1][:2] if keypoints[1][2] > 0.3 else keypoints[0][:2]
        shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2

        # Check if head is too far forward or back
        head_displacement = abs(head[0] - shoulder_center[0])
        shoulder_width = self.baseline_measurements.get('shoulder_width', 100)
        relative_displacement = head_displacement / shoulder_width

        # Adjusted threshold from 0.3 to 0.4
        if relative_displacement > 0.4:
            return FormFault(
                fault_type="neck_position",
                severity=FaultSeverity.MINOR,
                description="Poor neck alignment",
                frame=0,
                confidence=0.7,
                correction="Keep your neck in neutral position, look down at the floor"
            )
        return None

    def _check_hand_position(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check hand position relative to shoulders"""
        if not all(keypoints[i][2] > 0.3 for i in [5, 6, 9, 10]):
            return None

        # Check if hands are too far forward or back relative to shoulders
        shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2
        hand_center = (keypoints[9][:2] + keypoints[10][:2]) / 2

        hand_offset = abs(hand_center[0] - shoulder_center[0])
        shoulder_width = self.baseline_measurements.get('shoulder_width', 100)
        relative_offset = hand_offset / shoulder_width

        # Adjusted threshold from 0.4 to 0.5
        if relative_offset > 0.5:
            return FormFault(
                fault_type="hand_position",
                severity=FaultSeverity.MINOR,
                description="Hands positioned too far from shoulders",
                frame=0,
                confidence=0.7,
                correction="Position hands directly under your shoulders"
            )
        return None

    def _check_pushup_body_alignment(self, keypoints: np.ndarray, rep_state: str) -> Optional[FormFault]:
        """Check overall body alignment in pushups"""
        if not all(keypoints[i][2] > 0.3 for i in [5, 6, 11, 12, 15, 16]):
            return None

        # Calculate deviation from straight line (head to heels)
        shoulder_center = (keypoints[5][:2] + keypoints[6][:2]) / 2
        hip_center = (keypoints[11][:2] + keypoints[12][:2]) / 2
        ankle_center = (keypoints[15][:2] + keypoints[16][:2]) / 2

        # Check if hip deviates from shoulder-ankle line
        hip_deviation = self._point_line_distance(hip_center, shoulder_center, ankle_center)
        body_length = np.linalg.norm(shoulder_center - ankle_center)
        relative_deviation = hip_deviation / body_length

        # Adjusted threshold from 0.1 to 0.15
        if relative_deviation > 0.15:
            return FormFault(
                fault_type="body_alignment",
                severity=FaultSeverity.MODERATE,
                description="Poor body alignment",
                frame=0,
                confidence=0.9,
                correction="Maintain straight line from head to heels"
            )
        return None

    # Utility methods
    def _point_line_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """Calculate signed perpendicular distance from point to line"""
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        # Vector from line_start to point
        point_vec = point - line_start

        # Handle zero-length line
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-6:
            return float(np.linalg.norm(point_vec))

        # Project point onto line
        proj_length = np.dot(point_vec, line_vec) / line_len_sq
        proj = line_start + proj_length * line_vec

        # Calculate signed distance (positive = right side of line)
        to_point = point - proj
        perpendicular = np.array([-line_vec[1], line_vec[0]])  # Rotate 90 degrees
        perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)

        return np.dot(to_point, perpendicular)

    def get_form_summary(self) -> Dict:
        """Get comprehensive summary of form issues throughout the session"""
        if not self.fault_history:
            return {
                'total_faults': 0,
                'form_score': 100,
                'recommendations': ["Excellent form! No major issues detected."],
                'session_quality': 'excellent'
            }

        # Count faults by type and severity
        fault_counts = dict(self.fault_counts)
        severity_counts = {s: 0 for s in FaultSeverity}

        for fault in self.fault_history:
            severity_counts[fault.severity] += 1

        # Enhanced form score calculation with more reasonable weights
        severity_weights = {
            FaultSeverity.MINOR: 2,      # Reduced from 3
            FaultSeverity.MODERATE: 5,    # Reduced from 8
            FaultSeverity.SEVERE: 10      # Reduced from 15
        }

        # Penalize frequent faults more moderately
        total_deduction = 0
        for severity, count in severity_counts.items():
            base_deduction = count * severity_weights[severity]
            # More moderate frequency penalty
            frequency_multiplier = 1 + (count / 100)  # Changed from 50 to 100
            total_deduction += base_deduction * frequency_multiplier

        form_score = max(0, 100 - total_deduction)

        # Determine session quality
        if form_score >= 90:
            session_quality = 'excellent'
        elif form_score >= 75:
            session_quality = 'good'
        elif form_score >= 60:
            session_quality = 'fair'
        else:
            session_quality = 'needs_improvement'

        # Generate prioritized recommendations
        recommendations = []

        # Only process fault types that actually exist in fault_history
        valid_fault_types = set(f.fault_type for f in self.fault_history)
        valid_fault_counts = {k: v for k, v in fault_counts.items() if k in valid_fault_types}

        if valid_fault_counts:
            most_critical_faults = sorted(
                [(fault_type, count) for fault_type, count in valid_fault_counts.items()],
                key=lambda x: (
                    max([['minor', 'moderate', 'severe'].index(f.severity.value)
                         for f in self.fault_history if f.fault_type == x[0]]),
                    x[1]
                ),
                reverse=True
            )[:3]  # Top 3 most critical issues (reduced from 5)

            for fault_type, count in most_critical_faults:
                # Find the most severe instance and its correction
                fault_instances = [f for f in self.fault_history if f.fault_type == fault_type]
                if fault_instances:  # Double-check we have instances
                    most_severe = max(fault_instances,
                                      key=lambda f: ['minor', 'moderate', 'severe'].index(f.severity.value))

                    recommendations.append({
                        'priority': ['minor', 'moderate', 'severe'].index(most_severe.severity.value) + 1,
                        'fault_type': fault_type,
                        'description': most_severe.description,
                        'correction': most_severe.correction,
                        'frequency': count,
                        'severity': most_severe.severity.value
                    })

        # Calculate consistency metrics
        fault_distribution = len(valid_fault_counts) / max(1, len(self.fault_history))
        consistency_score = max(0, 100 - fault_distribution * 50)

        return {
            'total_faults': len(self.fault_history),
            'unique_fault_types': len(valid_fault_counts),
            'fault_counts': valid_fault_counts,
            'severity_breakdown': {s.value: severity_counts[s] for s in FaultSeverity},
            'form_score': round(form_score, 1),
            'consistency_score': round(consistency_score, 1),
            'session_quality': session_quality,
            'recommendations': recommendations,
            'fault_timeline': [
                {
                    'frame': f.frame,
                    'type': f.fault_type,
                    'severity': f.severity.value,
                    'description': f.description,
                    'confidence': f.confidence
                }
                for f in self.fault_history
            ],
            'summary_stats': {
                'avg_faults_per_100_frames': (len(self.fault_history) / max(1, self.frame_count)) * 100,
                'most_common_fault': max(valid_fault_counts.items(), key=lambda x: x[1])[
                    0] if valid_fault_counts else None,
                'severity_ratio': {
                    'minor_pct': (severity_counts[FaultSeverity.MINOR] / max(1, len(self.fault_history))) * 100,
                    'moderate_pct': (severity_counts[FaultSeverity.MODERATE] / max(1, len(self.fault_history))) * 100,
                    'severe_pct': (severity_counts[FaultSeverity.SEVERE] / max(1, len(self.fault_history))) * 100,
                }
            }
        }