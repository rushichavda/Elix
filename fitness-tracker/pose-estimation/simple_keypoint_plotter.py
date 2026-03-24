# Simple example to get you started quickly
from src.models.mediapipe_model import MediaPipeModel
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def simple_keypoint_analysis(video_path, max_frames=100):
    """Simple function to extract and plot keypoints from video"""
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None, None
    
    # Initialize model
    print("Initializing MediaPipe model...")
    try:
        model = MediaPipeModel(complexity=1, min_detection_confidence=0.5)
        print("Model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None, None
    
    # Storage for keypoints
    all_keypoints = []
    timestamps = []
    
    # Process video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Resolution: {width}x{height}")
    print(f"Processing up to {max_frames} frames...")
    
    frame_count = 0
    processed_count = 0
    
    while processed_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video reached at frame {frame_count}")
            break
        
        # Process frame with error handling
        try:
            results = model.process_frame(frame)
            keypoints = results['keypoints'][0]  # Shape: (33, 3)
            
            # Check if we got valid keypoints
            if keypoints.shape == (33, 3):
                all_keypoints.append(keypoints)
                timestamps.append(frame_count / fps)
                processed_count += 1
                
                if processed_count % 30 == 0:
                    print(f"Processed {processed_count} frames")
            else:
                print(f"Warning: Invalid keypoints shape {keypoints.shape} at frame {frame_count}")
                
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
        
        frame_count += 1
    
    cap.release()
    
    if len(all_keypoints) == 0:
        print("Error: No keypoints were extracted from the video")
        return None, None
    
    # Convert to numpy array
    all_keypoints = np.array(all_keypoints)  # Shape: (frames, 33, 3)
    timestamps = np.array(timestamps)
    
    print(f"Successfully extracted keypoints from {len(all_keypoints)} frames")
    print(f"Keypoints array shape: {all_keypoints.shape}")
    
    # Plot all coordinates for selected keypoints
    plot_selected_keypoints(all_keypoints, timestamps)
    
    return all_keypoints, timestamps

def plot_selected_keypoints(keypoints_data, timestamps):
    """Plot X, Y, Z coordinates for key body parts"""
    
    if keypoints_data is None or len(keypoints_data) == 0:
        print("No keypoints data to plot")
        return
    
    print(f"Plotting keypoints data with shape: {keypoints_data.shape}")
    
    # Define important keypoints to plot
    important_points = {
        # 'Nose': 0,
        # 'Left Shoulder': 11,
        # 'Right Shoulder': 12,
        # 'Left Elbow': 13,
        # 'Right Elbow': 14,
        'Left Wrist': 15,
        'Right Wrist': 16,
        'Left Hip': 23,
        'Right Hip': 24,
        'Left Knee': 25,
        'Right Knee': 26,
        # 'Left Ankle': 27,
        # 'Right Ankle': 28
    }
    
    # Create figure with subplots for each coordinate
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Keypoint Coordinates Over Time', fontsize=16)
    
    coordinate_names = ['X (pixels)', 'Y (pixels)', 'Visibility/Confidence']
    colors = plt.cm.tab10(np.linspace(0, 1, len(important_points)))
    
    for coord_idx, (ax, coord_name) in enumerate(zip(axes, coordinate_names)):
        for i, (point_name, kp_idx) in enumerate(important_points.items()):
            # Plot this coordinate for this keypoint over time
            values = keypoints_data[:, kp_idx, coord_idx]
            ax.plot(timestamps, values, label=point_name, 
                   color=colors[i], alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel(coord_name)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_all_keypoints_summary(keypoints_data, timestamps):
    """Create a summary plot showing all 33 keypoints"""
    
    if keypoints_data is None or len(keypoints_data) == 0:
        print("No keypoints data to plot in summary")
        return
    
    # Plot heatmap showing all keypoints over time
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('All 33 Keypoints Over Time', fontsize=16)
    
    coordinate_names = ['X Coordinates', 'Y Coordinates', 'Visibility Values']
    
    for coord_idx, (ax, coord_name) in enumerate(zip(axes, coordinate_names)):
        # Create heatmap for this coordinate
        data = keypoints_data[:, :, coord_idx].T  # Shape: (33, frames)
        
        im = ax.imshow(data, aspect='auto', cmap='viridis', 
                      extent=[timestamps[0], timestamps[-1], 0, 33])
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Keypoint Index')
        ax.set_title(coord_name)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()

# Usage example
if __name__ == "__main__":
    # Use the correct video path (replace with your actual video file)
    video_path = r"data\videos\good_form\squat_good_1.mp4"
    
    # Check if file exists and provide alternative paths
    if not os.path.exists(video_path):
        print(f"Video not found at: {video_path}")
        
        # Try some alternative paths
        alternative_paths = [
            "squat_good_1.mp4",
            "data/videos/good_form/squat_good_1.mp4",
            os.path.join("data", "videos", "good_form", "squat_good_1.mp4"),
            # Add your video path here
        ]
        
        video_found = False
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                video_path = alt_path
                video_found = True
                print(f"Found video at: {video_path}")
                break
        
        if not video_found:
            print("Please update the video_path variable with the correct path to your video file")
            print("Current working directory:", os.getcwd())
            print("Files in current directory:", os.listdir("."))
            exit(1)
    
    # Extract keypoints
    keypoints, timestamps = simple_keypoint_analysis(video_path, max_frames=200)
    
    # Create additional summary plot only if data was extracted
    if keypoints is not None and timestamps is not None:
        plot_all_keypoints_summary(keypoints, timestamps)
        print("Analysis completed successfully!")
    else:
        print("Analysis failed - no keypoints extracted")