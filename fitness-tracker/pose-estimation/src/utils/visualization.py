# Python file
# src/utils/visualization.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import seaborn as sns
from pathlib import Path

# Import 3D support - this is the key fix
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D

# Type checking imports
if TYPE_CHECKING:
    from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3DType
else:
    Axes3DType = type(None)

class PoseVisualizer:
    """Advanced visualization tools for pose analysis"""
    
    def __init__(self):
        # Color schemes
        self.colors = {
            'good': (0, 255, 0),      # Green
            'warning': (0, 165, 255),  # Orange
            'error': (0, 0, 255),      # Red
            'neutral': (255, 255, 255) # White
        }
        
        # Skeleton connections for different formats
        self.skeletons = {
            'coco': [
                [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                [2, 4], [3, 5], [4, 6], [5, 7]
            ],
            'mediapipe': [
                [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5],
                [5, 6], [6, 8], [9, 10], [11, 12], [11, 13],
                [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
                [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
                [18, 20], [11, 23], [12, 24], [23, 24], [23, 25],
                [24, 26], [25, 27], [26, 28], [27, 29], [28, 30],
                [29, 31], [30, 32], [27, 31], [28, 32]
            ]
        }
    
    def draw_enhanced_skeleton(self, frame: np.ndarray, keypoints: np.ndarray,
                             form_analysis: Optional[Dict] = None,
                             exercise_info: Optional[Dict] = None) -> np.ndarray:
        """Draw skeleton with enhanced visualization"""
        annotated = frame.copy()
        height, width = frame.shape[:2]
        
        # Create overlay for transparency effects
        overlay = frame.copy()
        
        # Draw skeleton
        self._draw_skeleton_connections(overlay, keypoints)
        
        # Draw keypoints with confidence visualization
        self._draw_keypoints_with_confidence(overlay, keypoints)
        
        # Add form analysis visualization
        if form_analysis:
            self._add_form_feedback(overlay, form_analysis, width, height)
        
        # Add exercise info
        if exercise_info:
            self._add_exercise_info(overlay, exercise_info, width, height)
        
        # Blend overlay with original
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
        
        return annotated
    
    def _draw_skeleton_connections(self, frame: np.ndarray, keypoints: np.ndarray):
        """Draw skeleton connections with gradient colors"""
        skeleton = self.skeletons['coco']
        
        for connection in skeleton:
            if len(keypoints.shape) == 3:  # Multiple people
                keypoints = keypoints[0]  # Use first person
            
            kpt1_idx, kpt2_idx = connection
            if kpt1_idx < len(keypoints) and kpt2_idx < len(keypoints):
                kpt1 = keypoints[kpt1_idx - 1]  # Adjust for 0-indexing
                kpt2 = keypoints[kpt2_idx - 1]
                
                if kpt1[2] > 0.5 and kpt2[2] > 0.5:  # Confidence check
                    # Calculate color based on confidence
                    avg_conf = (kpt1[2] + kpt2[2]) / 2
                    color = self._get_confidence_color(avg_conf)
                    
                    # Draw thicker line with gradient effect
                    cv2.line(frame, 
                            (int(kpt1[0]), int(kpt1[1])),
                            (int(kpt2[0]), int(kpt2[1])),
                            color, 3, cv2.LINE_AA)
    
    def _draw_keypoints_with_confidence(self, frame: np.ndarray, keypoints: np.ndarray):
        """Draw keypoints with size based on confidence"""
        if len(keypoints.shape) == 3:
            keypoints = keypoints[0]
        
        for i, kpt in enumerate(keypoints):
            if kpt[2] > 0.3:  # Low confidence threshold
                # Size based on confidence
                radius = int(5 + kpt[2] * 5)
                color = self._get_confidence_color(kpt[2])
                
                # Draw outer circle
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 
                          radius + 2, (0, 0, 0), -1)
                # Draw inner circle
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 
                          radius, color, -1)
                
                # Add keypoint number for debugging
                if kpt[2] > 0.7:
                    cv2.putText(frame, str(i), 
                               (int(kpt[0] - 5), int(kpt[1] - 5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                               (255, 255, 255), 1)
    
    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence score"""
        if confidence > 0.8:
            return self.colors['good']
        elif confidence > 0.6:
            return self.colors['warning']
        else:
            return self.colors['error']
    
    def _add_form_feedback(self, frame: np.ndarray, form_analysis: Dict,
                          width: int, height: int):
        """Add form feedback visualization"""
        # Create semi-transparent background for text
        feedback_area = np.zeros((150, width, 3), dtype=np.uint8)
        
        y_offset = 20
        for fault in form_analysis.get('faults', []):
            color = self.colors['error'] if fault.severity.value == 'severe' else self.colors['warning']
            
            # Add warning icon
            cv2.putText(feedback_area, "⚠", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add fault description
            cv2.putText(feedback_area, fault.description, (40, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            y_offset += 25
        
        # Add form score
        form_score = form_analysis.get('form_score', 100)
        score_color = self._get_score_color(form_score)
        cv2.putText(feedback_area, f"Form Score: {form_score}/100", 
                   (width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        
        # Blend feedback area with frame
        frame[height - 150:height, :] = cv2.addWeighted(
            frame[height - 150:height, :], 0.7,
            feedback_area, 0.3, 0
        )
    
    def _add_exercise_info(self, frame: np.ndarray, exercise_info: Dict,
                          width: int, height: int):
        """Add exercise information overlay"""
        # Top bar with exercise info
        info_bar = np.zeros((60, width, 3), dtype=np.uint8)
        
        # Exercise name
        cv2.putText(info_bar, exercise_info.get('exercise', 'Unknown').upper(),
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   self.colors['neutral'], 2)
        
        # Rep count
        rep_count = exercise_info.get('rep_count', 0)
        cv2.putText(info_bar, f"Reps: {rep_count}",
                   (width // 2 - 50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                   self.colors['good'], 2)
        
        # Current state
        state = exercise_info.get('state', 'neutral')
        state_color = self._get_state_color(state)
        cv2.putText(info_bar, f"State: {state.upper()}",
                   (width - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   state_color, 1)
        
        # Add angle info
        if 'primary_angle' in exercise_info:
            angle = exercise_info['primary_angle']
            cv2.putText(info_bar, f"Angle: {angle:.0f}°",
                       (width - 200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       self.colors['neutral'], 1)
        
        # Blend info bar
        frame[0:60, :] = cv2.addWeighted(frame[0:60, :], 0.7, info_bar, 0.3, 0)
    
    def _get_score_color(self, score: float) -> Tuple[int, int, int]:
        """Get color based on form score"""
        if score >= 80:
            return self.colors['good']
        elif score >= 60:
            return self.colors['warning']
        else:
            return self.colors['error']
    
    def _get_state_color(self, state: str) -> Tuple[int, int, int]:
        """Get color based on exercise state"""
        if state == 'down':
            return self.colors['error']
        elif state == 'up':
            return self.colors['good']
        else:
            return self.colors['neutral']
    
    def create_angle_plot(self, angle_history: List[Dict], save_path: Optional[str] = None):
        """Create detailed angle analysis plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        frames = list(range(len(angle_history)))
        
        # Plot 1: Primary angle with phases
        ax1 = axes[0, 0]
        angles = [a.get('primary_angle', 0) for a in angle_history]
        states = [a.get('state', 'neutral') for a in angle_history]
        
        # Color segments by state
        state_colors = {'up': 'green', 'down': 'red', 'neutral': 'gray'}
        for i in range(len(frames) - 1):
            ax1.plot(frames[i:i+2], angles[i:i+2], 
                    color=state_colors.get(states[i], 'gray'), linewidth=2)
        
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Angle (degrees)')
        ax1.set_title('Primary Joint Angle Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Velocity analysis
        ax2 = axes[0, 1]
        velocity = np.diff(angles)
        ax2.plot(frames[1:], velocity, 'b-', alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Angular Velocity (deg/frame)')
        ax2.set_title('Joint Angular Velocity')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Form score over time
        ax3 = axes[1, 0]
        form_scores = [a.get('form_score', 100) for a in angle_history]
        ax3.plot(frames, form_scores, 'g-', linewidth=2)
        ax3.fill_between(frames, form_scores, alpha=0.3, color='green')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Form Score')
        ax3.set_title('Form Quality Over Time')
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rep detection
        ax4 = axes[1, 1]
        rep_counts = [a.get('rep_count', 0) for a in angle_history]
        ax4.step(frames, rep_counts, 'r-', where='post', linewidth=2)
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Rep Count')
        ax4.set_title('Repetition Detection')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_comparison_dashboard(self, results: Dict, save_path: Optional[str] = None):
        """Create comprehensive comparison dashboard"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Model performance comparison
        ax1 = fig.add_subplot(gs[0, :2])
        models = list(results.keys())
        fps_values = [results[m].get('avg_fps', 0) for m in models]
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(models)))
        color_list = [tuple(c) for c in colors]
        
        bars = ax1.bar(models, fps_values, color=color_list)
        ax1.set_ylabel('Frames Per Second')
        ax1.set_title('Processing Speed Comparison')
        ax1.axhline(y=24, color='r', linestyle='--', alpha=0.5, label='Real-time threshold')
        
        # Add value labels
        for bar, fps in zip(bars, fps_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{fps:.1f}', ha='center', va='bottom')
        
        # Rep counting accuracy
        ax2 = fig.add_subplot(gs[0, 2])
        accuracies = [results[m].get('rep_accuracy', 0) for m in models]
        ax2.pie(accuracies, labels=models, autopct='%1.1f%%', colors=color_list)
        ax2.set_title('Rep Counting Accuracy')
        
        # Form detection comparison
        ax3 = fig.add_subplot(gs[1, :])
        form_data = []
        for model in models:
            form_data.append([
                results[model].get('form_score', 0),
                results[model].get('faults_detected', 0),
                results[model].get('false_positives', 0)
            ])
        
        form_data = np.array(form_data).T
        x = np.arange(len(models))
        width = 0.25
        
        ax3.bar(x - width, form_data[0], width, label='Form Score', color='green', alpha=0.7)
        ax3.bar(x, form_data[1], width, label='Faults Detected', color='orange', alpha=0.7)
        ax3.bar(x + width, form_data[2], width, label='False Positives', color='red', alpha=0.7)
        
        ax3.set_xlabel('Models')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.set_title('Form Analysis Performance')
        
        # Detailed metrics table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create table data
        metrics = ['FPS', 'CPU Usage', 'Memory', 'Accuracy', 'Form Score']
        table_data = []
        for model in models:
            row = [
                f"{results[model].get('avg_fps', 0):.1f}",
                f"{results[model].get('cpu_usage', 0):.0f}%",
                f"{results[model].get('memory_mb', 0):.0f}MB",
                f"{results[model].get('rep_accuracy', 0):.1f}%",
                f"{results[model].get('form_score', 0):.0f}"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data, rowLabels=models, colLabels=metrics,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color cells based on performance
        for i in range(len(models)):
            for j in range(len(metrics)):
                cell = table[(i+1, j)]
                if j == 0:  # FPS
                    val = float(table_data[i][j])
                    if val >= 24:
                        cell.set_facecolor('#90EE90')
                    elif val >= 15:
                        cell.set_facecolor('#FFD700')
                    else:
                        cell.set_facecolor('#FFB6C1')
        
        plt.suptitle('Pose Estimation Model Comparison Dashboard', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_3d_pose_visualization(self, keypoints_3d: np.ndarray, save_path: Optional[str] = None):
        """Create 3D pose visualization (for MediaPipe world landmarks)"""
        
        fig = plt.figure(figsize=(10, 10))
        
        # Create 3D subplot and explicitly cast to Axes3D
        ax = fig.add_subplot(111, projection='3d')
        
        # Alternative approach that ensures proper 3D axes methods
        # This helps with type checking and ensures all 3D methods are available
        if not hasattr(ax, 'set_zlabel'):
            # Force recreation as Axes3D if needed
            ax = Axes3D(fig)
        
        # MediaPipe connections for 3D visualization
        connections = [
            [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5],
            [5, 6], [6, 8], [9, 10], [11, 12], [11, 13],
            [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
            [23, 24], [23, 25], [24, 26], [25, 27], [26, 28]
        ]
        
        # Plot connections
        for connection in connections:
            if connection[0] < len(keypoints_3d) and connection[1] < len(keypoints_3d):
                pts = keypoints_3d[[connection[0], connection[1]]]
                # Use the generic plot method for 3D
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=2)
        
        # Plot keypoints
        ax.scatter(keypoints_3d[:, 0], keypoints_3d[:, 1], keypoints_3d[:, 2],
                    c='red', s=50)
        
        # Set labels using try-except to handle any type checking issues
        try:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')  # type: ignore
        except AttributeError:
            # Fallback method if set_zlabel is not recognized
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.zaxis.set_label_text('Z')
        
        ax.set_title('3D Pose Visualization')
        
        # Set equal aspect ratio
        max_range = np.array([keypoints_3d[:, 0].max() - keypoints_3d[:, 0].min(),
                             keypoints_3d[:, 1].max() - keypoints_3d[:, 1].min(),
                             keypoints_3d[:, 2].max() - keypoints_3d[:, 2].min()]).max() / 2.0
        
        mid_x = (keypoints_3d[:, 0].max() + keypoints_3d[:, 0].min()) * 0.5
        mid_y = (keypoints_3d[:, 1].max() + keypoints_3d[:, 1].min()) * 0.5
        mid_z = (keypoints_3d[:, 2].max() + keypoints_3d[:, 2].min()) * 0.5
        
        # Set limits using try-except for z-axis
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        try:
            ax.set_zlim(mid_z - max_range, mid_z + max_range)  # type: ignore
        except AttributeError:
            # Fallback method
            ax.set_zlim3d(mid_z - max_range, mid_z + max_range)
        
        # Alternative: Set box aspect for equal scaling (matplotlib >= 3.3.0)
        try:
            ax.set_box_aspect([1,1,1])
        except AttributeError:
            pass  # Older matplotlib versions don't have this method
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_3d_pose_animation(self, keypoints_history: List[np.ndarray], 
                                save_path: Optional[str] = None, 
                                fps: int = 30):
        """Create animated 3D pose visualization"""
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Ensure we have Axes3D functionality
        if not hasattr(ax, 'set_zlabel'):
            ax = Axes3D(fig)
        
        # MediaPipe connections
        connections = [
            [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5],
            [5, 6], [6, 8], [9, 10], [11, 12], [11, 13],
            [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
            [23, 24], [23, 25], [24, 26], [25, 27], [26, 28]
        ]
        
        # Initialize plot elements
        lines = []
        for _ in connections:
            line, = ax.plot([], [], [], 'b-', linewidth=2)
            lines.append(line)
        
        scatter = ax.scatter([], [], [], c='red', s=50)
        
        # Set labels with fallback
        try:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')  # type: ignore
        except AttributeError:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.zaxis.set_label_text('Z')
        
        ax.set_title('3D Pose Animation')
        
        # Calculate overall limits
        all_keypoints = np.concatenate(keypoints_history, axis=0)
        max_range = np.array([all_keypoints[:, 0].max() - all_keypoints[:, 0].min(),
                             all_keypoints[:, 1].max() - all_keypoints[:, 1].min(),
                             all_keypoints[:, 2].max() - all_keypoints[:, 2].min()]).max() / 2.0
        
        mid_x = (all_keypoints[:, 0].max() + all_keypoints[:, 0].min()) * 0.5
        mid_y = (all_keypoints[:, 1].max() + all_keypoints[:, 1].min()) * 0.5
        mid_z = (all_keypoints[:, 2].max() + all_keypoints[:, 2].min()) * 0.5
        
        # Set limits with fallback
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        try:
            ax.set_zlim(mid_z - max_range, mid_z + max_range)  # type: ignore
        except AttributeError:
            ax.set_zlim3d(mid_z - max_range, mid_z + max_range)
        
        def update(frame):
            keypoints = keypoints_history[frame]
            
            # Update lines
            for i, (connection, line) in enumerate(zip(connections, lines)):
                if connection[0] < len(keypoints) and connection[1] < len(keypoints):
                    pts = keypoints[[connection[0], connection[1]]]
                    line.set_data(pts[:, 0], pts[:, 1])
                    line.set_3d_properties(pts[:, 2])
            
            # Update scatter plot
            scatter._offsets3d = (keypoints[:, 0], keypoints[:, 1], keypoints[:, 2])
            
            return lines + [scatter]
        
        anim = FuncAnimation(fig, update, frames=len(keypoints_history), 
                           interval=1000/fps, blit=True)
        
        if save_path:
            anim.save(save_path, fps=fps, extra_args=['-vcodec', 'libx264'])
        else:
            plt.show()
        
        plt.close()

# Example usage function to test the 3D visualization
def test_3d_visualization():
    """Test function to verify 3D visualization works without z-axis errors"""
    visualizer = PoseVisualizer()
    
    # Create dummy 3D keypoints (33 MediaPipe landmarks)
    keypoints_3d = np.random.randn(33, 3)
    
    # This should work without any z-axis errors
    visualizer.create_3d_pose_visualization(keypoints_3d, save_path='test_3d_pose.png')
    print("3D visualization created successfully!")

if __name__ == "__main__":
    test_3d_visualization()