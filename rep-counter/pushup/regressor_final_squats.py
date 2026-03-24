import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
import threading
import time

# Import wave pipeline functions
from wave_pipeline import reduce_noise_iterative, segment_waves, categorize_wave, detect_waves_streaming_simple

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,  # Reduced complexity for better performance
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables for tracking state
current_arm: str = "right"  # Currently tracked arm (left or right)

# Wave-based tracking variables
state_sequence_list: list[int] = []  # Store state sequence (1-5)
denoised_state_sequence: list[int] = []  # Store denoised state sequence
current_wave_state: int | None = None  # Current state (1-5)
wave_repetition_counter: int = 0  # Wave-based rep counter
detected_wave_segments: list[tuple] = []  # Store detected wave segments
wave_rep_quality_categories: list[str] = []  # Store wave-based rep categories

# Plotting and visualization variables - EXACTLY SAME AS SQUATS
plot_values: deque = deque(maxlen=1000)  # Store values for real-time plotting
plot_thread: threading.Thread | None = None
is_plot_running: bool = False

# Video timing control variables
video_frames_per_second: float = 30
target_frame_duration: float = 1.0 / 30
last_frame_timestamp: float = 0
frame_skip_counter: int = 0


class PushupRegressor(nn.Module):
    """
    Neural network model for pushup position regression.
    
    Takes normalized keypoint coordinates as input and outputs a continuous
    value between 0-1 representing pushup depth/position.
    """
    
    def __init__(self, input_feature_count: int = 4) -> None:
        """
        Initialize the pushup regression model.
        
        Args:
            input_feature_count: Number of input features (default: 4 normalized keypoints)
        """
        super(PushupRegressor, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(input_feature_count, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.
        
        Args:
            input_tensor: Input tensor of shape (batch_size, input_feature_count)
            
        Returns:
            Output tensor with sigmoid activation (values between 0-1)
        """
        x = self.dropout(self.relu(self.fc1(input_tensor)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        return torch.sigmoid(self.output(x))


def normalize_keypoint_coordinates(raw_keypoints: list[float]) -> list[float]:
    """
    Normalize keypoint coordinates relative to elbow position.
    
    This reduces the impact of camera distance and person size by making
    all measurements relative to the elbow joint.
    
    Args:
        raw_keypoints: List of 6 values [wrist_x, wrist_y, elbow_x, elbow_y, shoulder_x, shoulder_y]
        
    Returns:
        List of 4 normalized values relative to elbow position
    """
    # Extract elbow coordinates as reference point
    elbow_x_coordinate, elbow_y_coordinate = raw_keypoints[2], raw_keypoints[3]
    
    return [
        raw_keypoints[0] - elbow_x_coordinate,  # Wrist x relative to elbow
        raw_keypoints[1] - elbow_y_coordinate,  # Wrist y relative to elbow
        raw_keypoints[4] - elbow_x_coordinate,  # Shoulder x relative to elbow
        raw_keypoints[5] - elbow_y_coordinate   # Shoulder y relative to elbow
    ]


def draw_selected_arm_landmarks(frame: cv2.Mat, pose_results: mp.solutions.pose.Pose) -> None:
    """
    Draw pose landmarks and connections for the currently selected arm.
    
    Only draws the arm with better visibility to reduce visual clutter.
    
    Args:
        frame: OpenCV frame to draw on
        pose_results: MediaPipe pose detection results
    """
    global current_arm
    
    if not pose_results.pose_landmarks:
        return

    landmarks = pose_results.pose_landmarks.landmark
    
    # MediaPipe landmark indices for right and left arm joints
    RIGHT_ARM_INDICES = [12, 14, 16]  # [shoulder, elbow, wrist]
    LEFT_ARM_INDICES = [11, 13, 15]   # [shoulder, elbow, wrist]
    
    selected_arm_indices = RIGHT_ARM_INDICES if current_arm == "right" else LEFT_ARM_INDICES

    # Check if all landmarks are visible enough to draw
    if not all(landmarks[i].visibility > 0.5 for i in selected_arm_indices):
        return

    # Draw keypoints as circles
    for landmark_index in selected_arm_indices:
        x_pixel = int(landmarks[landmark_index].x * frame.shape[1])
        y_pixel = int(landmarks[landmark_index].y * frame.shape[0])
        cv2.circle(frame, (x_pixel, y_pixel), 5, (0, 255, 0), -1)
    
    # Draw connections between joints
    arm_connections = [(selected_arm_indices[0], selected_arm_indices[1]), 
                       (selected_arm_indices[1], selected_arm_indices[2])]
    
    for start_joint_index, end_joint_index in arm_connections:
        start_x = int(landmarks[start_joint_index].x * frame.shape[1])
        start_y = int(landmarks[start_joint_index].y * frame.shape[0])
        end_x = int(landmarks[end_joint_index].x * frame.shape[1])
        end_y = int(landmarks[end_joint_index].y * frame.shape[0])
        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)


def extract_and_select_best_arm_keypoints(pose_results: mp.solutions.pose.Pose) -> list[float] | None:
    """
    Extract keypoints from the arm with better visibility.
    
    Automatically switches between left and right arm based on visibility scores
    to ensure consistent tracking even when one arm is occluded.
    
    Args:
        pose_results: MediaPipe pose detection results
        
    Returns:
        List of normalized keypoint coordinates or None if no valid arm found
    """
    global current_arm
    
    if not pose_results.pose_landmarks:
        return None

    landmarks = pose_results.pose_landmarks.landmark
    
    # MediaPipe landmark indices
    RIGHT_ARM_INDICES = [12, 14, 16]  # [shoulder, elbow, wrist]
    LEFT_ARM_INDICES = [11, 13, 15]   # [shoulder, elbow, wrist]

    # Calculate visibility scores for both arms
    right_arm_visibility = [landmarks[i].visibility for i in RIGHT_ARM_INDICES]
    left_arm_visibility = [landmarks[i].visibility for i in LEFT_ARM_INDICES]

    # Check which arms have sufficient visibility
    is_right_arm_valid = all(visibility > 0.5 for visibility in right_arm_visibility)
    is_left_arm_valid = all(visibility > 0.5 for visibility in left_arm_visibility)

    # Return None if neither arm is sufficiently visible
    if not is_right_arm_valid and not is_left_arm_valid:
        return None

    # Calculate total visibility scores for arm switching logic
    right_arm_total_visibility = sum(right_arm_visibility)
    left_arm_total_visibility = sum(left_arm_visibility)

    # Switch to better arm if visibility difference is significant
    visibility_threshold = 0.05
    if current_arm == "right" and is_left_arm_valid and (left_arm_total_visibility - right_arm_total_visibility) > visibility_threshold:
        current_arm = "left"
    elif current_arm == "left" and is_right_arm_valid and (right_arm_total_visibility - left_arm_total_visibility) > visibility_threshold:
        current_arm = "right"

    # Extract coordinates for selected arm
    selected_arm_indices = RIGHT_ARM_INDICES if current_arm == "right" else LEFT_ARM_INDICES
    raw_keypoints = [
        landmarks[selected_arm_indices[2]].x, landmarks[selected_arm_indices[2]].y,  # Wrist
        landmarks[selected_arm_indices[1]].x, landmarks[selected_arm_indices[1]].y,  # Elbow
        landmarks[selected_arm_indices[0]].x, landmarks[selected_arm_indices[0]].y   # Shoulder
    ]
    
    return normalize_keypoint_coordinates(raw_keypoints)


def classify_pushup_state_using_waves(model: PushupRegressor, normalized_keypoints: list[float]) -> float:
    """
    Classify current pushup state using wave-based detection method.
    
    Processes keypoints through the neural network and updates the state sequence
    for wave pattern detection and rep counting.
    
    Args:
        model: Trained pushup regression model
        normalized_keypoints: List of normalized keypoint coordinates
        
    Returns:
        Continuous value between 0-1 representing pushup depth
    """
    model.eval()
    with torch.no_grad():
        # Convert to tensor and get model prediction
        input_tensor = torch.tensor(normalized_keypoints, dtype=torch.float32).unsqueeze(0)
        model_output = model(input_tensor)
        pushup_depth_value = model_output.item()
        
        # Update state sequence for wave detection
        update_state_sequence_for_wave_detection(pushup_depth_value)
        
        # Attempt wave-based rep detection if enough data available
        if len(state_sequence_list) >= 5:
            detect_repetitions_using_wave_analysis()
        
        return pushup_depth_value


def convert_continuous_value_to_discrete_state(continuous_value: float) -> int:
    """
    Convert continuous pushup depth value to discrete state.
    
    Maps the 0-1 continuous output to 5 discrete states representing
    different pushup positions from up to down.
    
    Args:
        continuous_value: Continuous value between 0-1
        
    Returns:
        Discrete state value between 1-5
    """
    if 0.0 <= continuous_value < 0.2:
        return 1  # Up/highest position
    elif 0.2 <= continuous_value < 0.4:
        return 2  # Quarter down
    elif 0.4 <= continuous_value < 0.6:
        return 3  # Half down
    elif 0.6 <= continuous_value < 0.8:
        return 4  # Three-quarter down
    elif 0.8 <= continuous_value <= 1.0:
        return 5  # Down/lowest position
    else:
        return 1  # Default to up for invalid values


def update_state_sequence_for_wave_detection(pushup_depth_value: float) -> None:
    """
    Update the state sequence used for wave pattern detection.
    
    Only adds new states when they change to create a sequence of discrete
    transitions suitable for wave analysis.
    
    Args:
        pushup_depth_value: Current continuous pushup depth value
    """
    global state_sequence_list, current_wave_state
    
    new_discrete_state = convert_continuous_value_to_discrete_state(pushup_depth_value)
    
    # Only add to sequence when state changes to avoid redundant data
    if current_wave_state != new_discrete_state:
        state_sequence_list.append(new_discrete_state)
        current_wave_state = new_discrete_state
        
        # Maintain manageable sequence length to prevent memory issues
        max_sequence_length = 100
        if len(state_sequence_list) > max_sequence_length:
            state_sequence_list = state_sequence_list[-80:]
            
            # Clean up old wave segments to maintain performance
            max_segments = 20
            if len(detected_wave_segments) > max_segments:
                detected_wave_segments[:] = detected_wave_segments[-15:]


def detect_repetitions_using_wave_analysis() -> list:
    """
    Detect complete pushup repetitions using wave pattern analysis.
    
    Uses signal processing techniques to identify complete pushup cycles
    and categorize their quality based on wave characteristics.
    
    Returns:
        List of detected segments (currently returns empty list)
    """
    global state_sequence_list, detected_wave_segments, wave_repetition_counter
    global wave_rep_quality_categories, denoised_state_sequence
    
    # Need minimum sequence length for meaningful analysis
    minimum_sequence_length = 5
    if len(state_sequence_list) < minimum_sequence_length:
        return []
    
    try:
        # Apply noise reduction and segment wave patterns
        denoised_sequence = reduce_noise_iterative(state_sequence_list.copy())
        detected_segments = segment_waves(denoised_sequence)
        
        # Process each detected segment with duplicate prevention
        for segment in detected_segments:
            start_index, end_index, wave_category = segment
            
            # Only process stable segments (not at the very end of sequence)
            if end_index < len(denoised_sequence) - 1:
                
                # Check for duplicate segments using position tolerance
                position_tolerance = 2
                is_duplicate_segment = any(
                    abs(start_index - existing_segment[0]) <= position_tolerance and 
                    abs(end_index - existing_segment[1]) <= position_tolerance 
                    for existing_segment in detected_wave_segments[-10:]
                )
                
                if not is_duplicate_segment:
                    # Add new valid segment
                    detected_wave_segments.append(segment)
                    wave_repetition_counter += 1
                    
                    # Map wave categories to quality ratings
                    quality_mapping = {
                        1: "GREEN",   # Full wave pattern - excellent form
                        2: "YELLOW",  # Partial wave pattern - acceptable form  
                    }
                    repetition_quality = quality_mapping.get(wave_category, "RED")  # Default to poor for other categories
                        
                    wave_rep_quality_categories.append(repetition_quality)
                    
                    # Log detection results
                    print(f"Rep {wave_repetition_counter}: {repetition_quality} (Category {wave_category})")
                    print(f"  Wave pattern: {denoised_sequence[start_index:end_index+1]}")
        
        # Store denoised sequence for potential future analysis
        denoised_state_sequence = denoised_sequence
        
    except Exception as error:
        print(f"Wave detection error: {error}")
    
    return []


def create_live_plot_visualization() -> None:
    """
    Create and manage live plotting of pushup depth values.
    
    EXACT COPY FROM SQUATS FILE - same buffer sizes, same display window, same everything
    """
    global plot_values, is_plot_running, state_sequence_list
    
    # Enable interactive plotting
    plt.ion()
    figure, axis = plt.subplots(1, 1, figsize=(8, 4))
    
    # Configure plot appearance - EXACTLY SAME AS SQUATS
    axis.set_ylim(0, 1)
    axis.set_xlim(0, 200)
    axis.set_xlabel('Frame')
    axis.set_ylabel('Value (0-1)')
    axis.set_title('Live Pushup Position Value')
    axis.grid(True)
    
    # Add horizontal lines showing state boundaries - EXACTLY SAME AS SQUATS
    axis.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='State 5 (0.8)')
    axis.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='State 4 (0.6)')
    axis.axhline(y=0.4, color='yellow', linestyle='--', alpha=0.5, label='State 3 (0.4)')
    axis.axhline(y=0.2, color='g', linestyle='--', alpha=0.7, label='State 2 (0.2)')
    axis.legend()
    
    # Initialize empty line for data
    data_line, = axis.plot([], [], 'b-', linewidth=2)
    
    # Control update frequency - EXACTLY SAME AS SQUATS
    plot_update_interval = 0.2
    last_update_time = time.time()
    
    while is_plot_running:
        try:
            current_time = time.time()
            
            # Update plot at controlled intervals
            if current_time - last_update_time >= plot_update_interval:
                if len(plot_values) > 0:
                    # Prepare data for last 200 points - EXACTLY SAME AS SQUATS
                    display_window_size = 200
                    x_coordinates = list(range(max(0, len(plot_values) - display_window_size), len(plot_values)))
                    y_coordinates = list(plot_values)[-display_window_size:]
                    
                    # Update line data and axis limits
                    data_line.set_data(x_coordinates, y_coordinates)
                    axis.set_xlim(max(0, len(plot_values) - display_window_size), len(plot_values))
                    
                    # Color line based on current pushup state - EXACTLY SAME AS SQUATS
                    if len(y_coordinates) > 0:
                        current_value = y_coordinates[-1]
                        current_state = convert_continuous_value_to_discrete_state(current_value)
                        state_colors = ['green', 'blue', 'cyan', 'orange', 'red']
                        data_line.set_color(state_colors[current_state - 1])
                
                # Refresh display
                figure.canvas.draw_idle()
                figure.canvas.flush_events()
                
                last_update_time = current_time
            
            # Brief sleep to prevent excessive CPU usage - EXACTLY SAME AS SQUATS
            time.sleep(0.1)
            
        except Exception as plotting_error:
            print(f"Plotting error: {plotting_error}")
            time.sleep(0.2)
    
    plt.close()


def start_live_plotting_thread() -> None:
    """
    Initialize and start the live plotting in a separate thread.
    
    Creates a daemon thread so it automatically closes when main program exits.
    """
    global plot_thread, is_plot_running
    
    is_plot_running = True
    plot_thread = threading.Thread(target=create_live_plot_visualization)
    plot_thread.daemon = True
    plot_thread.start()


def stop_live_plotting_thread() -> None:
    """
    Stop the live plotting thread and clean up resources.
    
    Safely terminates the plotting thread and waits for it to finish.
    """
    global is_plot_running
    
    is_plot_running = False
    if plot_thread:
        plot_thread.join()


def control_video_playback_timing(video_source: int | str) -> None:
    """
    Control video playback timing to maintain consistent frame rates.
    
    Implements frame skipping and timing control for both high FPS videos
    and normal playback to ensure smooth analysis.
    
    Args:
        video_source: Video source identifier (0 for camera, string for file)
    """
    global video_frames_per_second, target_frame_duration, last_frame_timestamp, frame_skip_counter
    
    # Camera feeds don't need timing control
    if video_source == 0:
        return
    
    current_time = time.time()
    
    # Implement frame skipping for high FPS videos
    high_fps_threshold = 40
    if video_frames_per_second > high_fps_threshold:
        frame_skip_counter += 1
        if frame_skip_counter % 2 == 0:
            return
    
    # Control timing based on target frame duration
    if last_frame_timestamp > 0:
        elapsed_time = current_time - last_frame_timestamp
        required_sleep_time = target_frame_duration - elapsed_time
        
        if required_sleep_time > 0:
            time.sleep(required_sleep_time)
    
    last_frame_timestamp = current_time


def run_pushup_analysis_inference(video_source: int | str = 0, model_file_path: str = "pushups_dataset_sigmoid.pkl") -> None:
    """
    Main inference function for wave-based pushup analysis.
    
    Handles both live camera feed and video file input, provides real-time
    analysis with rep counting and quality assessment.
    
    Args:
        video_source: Video input source (0 for camera, file path for video)
        model_file_path: Path to trained model weights file
    """
    global current_arm, wave_repetition_counter, video_frames_per_second, target_frame_duration, last_frame_timestamp
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pushup_model = PushupRegressor()
    pushup_model.load_state_dict(torch.load(model_file_path, map_location=device))
    pushup_model.to(device)
    pushup_model.eval()

    # Initialize video capture
    video_capture = cv2.VideoCapture(video_source)
    
    # Configure video source parameters
    if video_source == 0:  # Live camera configuration
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        video_frames_per_second = 30
        target_fps = 30
        print("Using live camera")
    else:  # Video file configuration
        video_frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
        if video_frames_per_second <= 0:
            video_frames_per_second = 30
        print(f"Original Video FPS: {video_frames_per_second}")
        
        # Adjust target FPS for high frame rate videos
        high_fps_threshold = 40
        if video_frames_per_second > high_fps_threshold:
            target_fps = 30
            print(f"High FPS detected. Targeting {target_fps} FPS for playback.")
        else:
            target_fps = video_frames_per_second
        
        target_frame_duration = 1.0 / target_fps
    
    last_frame_timestamp = time.time()
    
    # Verify video source is accessible
    if not video_capture.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    # Display initialization information
    print(f"Video source: {video_source}")
    print(f"Target playback FPS: {target_fps}")
    print("WAVE-BASED DETECTION SYSTEM ACTIVE")
    print("Controls: 'q' = quit, 's' = switch between camera/video")
    
    # Start real-time plotting
    start_live_plotting_thread()
    
    # Initialize frame processing counters
    total_frame_count = 0
    processed_frame_count = 0
    process_start_time = time.time()
    fps_update_interval = 1.0
    last_fps_update_time = time.time()
    display_fps = 0.0

    # Main processing loop
    while video_capture.isOpened():
        frame_read_success, current_frame = video_capture.read()
        
        # Handle end of video or camera errors
        if not frame_read_success:
            if video_source == 0:
                print("Camera error")
                break
            else:
                print("Video ended")
                break

        total_frame_count += 1
        
        # Skip frames for high FPS videos to maintain performance
        if video_source != 0 and video_frames_per_second > 40:
            if total_frame_count % 2 != 0:
                continue
        
        processed_frame_count += 1

        # Process current frame for pose detection
        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        pose_detection_results = pose.process(frame_rgb)

        # Analyze pose if landmarks detected
        if pose_detection_results.pose_landmarks:
            draw_selected_arm_landmarks(current_frame, pose_detection_results)
            extracted_keypoints = extract_and_select_best_arm_keypoints(pose_detection_results)
            
            if extracted_keypoints is not None:
                pushup_depth_value = classify_pushup_state_using_waves(pushup_model, extracted_keypoints)
                if pushup_depth_value is not None:
                    plot_values.append(pushup_depth_value)

        # Prepare frame display information
        frame_height, frame_width = current_frame.shape[:2]
        
        # Create background for text overlay
        cv2.rectangle(current_frame, (10, 10), (380, 100), (0, 0, 0), -1)
        
        # Display repetition counter
        cv2.putText(current_frame, f"Reps: {wave_repetition_counter}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Display quality statistics
        quality_statistics = get_repetition_quality_statistics()
        cv2.putText(current_frame, f"Good: {quality_statistics['GREEN']}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(current_frame, f"Fair: {quality_statistics['YELLOW']}", (130, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(current_frame, f"Poor: {quality_statistics['RED']}", (240, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display current state and value
        if len(plot_values) > 0:
            current_pushup_value = plot_values[-1]
            current_discrete_state = convert_continuous_value_to_discrete_state(current_pushup_value)
            cv2.putText(current_frame, f"State: {current_discrete_state} ({current_pushup_value:.3f})", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Calculate and display FPS information
        current_time = time.time()
        if current_time - last_fps_update_time >= fps_update_interval:
            elapsed_processing_time = current_time - process_start_time
            display_fps = processed_frame_count / elapsed_processing_time if elapsed_processing_time > 0 else 0
            last_fps_update_time = current_time
        
        # Color FPS display based on performance
        fps_tolerance = 5
        fps_display_color = (0, 255, 0) if abs(display_fps - target_fps) < fps_tolerance else (0, 255, 255)
        cv2.putText(current_frame, f"FPS: {display_fps:.1f}/{target_fps:.1f}", 
                   (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_display_color, 2)

        # Display video source information
        source_label = "Live Camera" if video_source == 0 else "Video File"
        cv2.putText(current_frame, source_label, (frame_width - 200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show processed frame
        cv2.imshow('Pushup Analysis - Wave Detection', current_frame)
        
        # Control playback timing
        control_video_playback_timing(video_source)
        
        # Handle keyboard input
        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('s'):
            # Switch between camera and video file
            video_capture.release()
            
            if video_source == 0:
                video_source = "The Perfect Push Up _ Do it right!.mp4"
            else:
                video_source = 0
            
            video_capture = cv2.VideoCapture(video_source)
            
            # Reset timing parameters for new source
            if video_source == 0:
                video_frames_per_second = 30
                target_fps = 30
                print("Switched to live camera")
            else:
                video_frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
                if video_frames_per_second <= 0:
                    video_frames_per_second = 30
                target_fps = 30 if video_frames_per_second > 40 else video_frames_per_second
                print(f"Switched to video file (FPS: {video_frames_per_second})")
            
            target_frame_duration = 1.0 / target_fps
            total_frame_count = 0
            processed_frame_count = 0
            process_start_time = time.time()
            last_frame_timestamp = time.time()

    # Cleanup resources
    video_capture.release()
    cv2.destroyAllWindows()
    stop_live_plotting_thread()
    
    # Display final analysis results
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total Reps: {wave_repetition_counter}")
    final_quality_stats = get_repetition_quality_statistics()
    print(f"Good: {final_quality_stats['GREEN']}")
    print(f"Fair: {final_quality_stats['YELLOW']}")
    print(f"Poor: {final_quality_stats['RED']}")


def get_most_recent_repetition_quality() -> str:
    """
    Get the quality category of the most recently completed repetition.
    
    Returns:
        Quality category string ("GREEN", "YELLOW", "RED", or "NONE")
    """
    if wave_rep_quality_categories:
        return wave_rep_quality_categories[-1]
    return "NONE"


def get_repetition_quality_statistics() -> dict[str, int]:
    """
    Calculate statistics for all completed repetitions by quality category.
    
    Returns:
        Dictionary with counts for each quality category
    """
    quality_counts = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    
    for quality_category in wave_rep_quality_categories:
        if quality_category in quality_counts:
            quality_counts[quality_category] += 1
            
    return quality_counts


if __name__ == "__main__":
    # Main execution - default to live camera
    # run_pushup_analysis_inference(0)  # Use live camera
    run_pushup_analysis_inference("The Perfect Push Up _ Do it right!.mp4")  # Alternative: use video file
    