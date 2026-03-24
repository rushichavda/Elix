import cv2
import numpy as np
import os
import sys

def test_video_reading():
    """Test if we can read the video file"""
    print("=== Testing Video Reading ===")
    
    video_paths = [
        r"data\videos\good_form\squat_good_1.mp4",
        "data/videos/good_form/squat_good_1.mp4",
        "squat_good_1.mp4"
    ]
    
    for video_path in video_paths:
        print(f"\nTrying: {video_path}")
        if os.path.exists(video_path):
            print(f"✓ File exists")
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"✓ Video opened successfully")
                print(f"  FPS: {fps}")
                print(f"  Total frames: {total_frames}")
                print(f"  Resolution: {width}x{height}")
                
                # Try to read first frame
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Successfully read first frame, shape: {frame.shape}")
                    cap.release()
                    return video_path, frame
                else:
                    print("✗ Could not read first frame")
                cap.release()
            else:
                print("✗ Could not open video")
        else:
            print("✗ File does not exist")
    
    print("\nNo valid video found. Current directory contents:")
    try:
        files = os.listdir(".")
        for f in files[:10]:  # Show first 10 files
            print(f"  {f}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
    except:
        print("Could not list directory contents")
    
    return None, None

def test_mediapipe_import():
    """Test if MediaPipe model can be imported and initialized"""
    print("\n=== Testing MediaPipe Import ===")
    
    try:
        print("Importing MediaPipe...")
        import mediapipe as mp
        print("✓ MediaPipe imported successfully")
        
        print("Creating MediaPipe Pose...")
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✓ MediaPipe Pose created successfully")
        return pose
        
    except ImportError as e:
        print(f"✗ Could not import MediaPipe: {e}")
        return None
    except Exception as e:
        print(f"✗ Error creating MediaPipe Pose: {e}")
        return None

def test_custom_model_import():
    """Test if custom MediaPipe model can be imported"""
    print("\n=== Testing Custom Model Import ===")
    
    try:
        print("Importing custom MediaPipeModel...")
        from src.models.mediapipe_model import MediaPipeModel
        print("✓ Custom MediaPipeModel imported successfully")
        
        print("Creating MediaPipeModel instance...")
        model = MediaPipeModel(complexity=1, min_detection_confidence=0.5)
        print("✓ MediaPipeModel instance created successfully")
        return model
        
    except ImportError as e:
        print(f"✗ Could not import custom MediaPipeModel: {e}")
        print("Make sure the src/models/mediapipe_model.py file exists")
        return None
    except Exception as e:
        print(f"✗ Error creating MediaPipeModel: {e}")
        return None

def test_basic_processing(video_path, frame, pose_model=None, custom_model=None):
    """Test basic frame processing"""
    print("\n=== Testing Frame Processing ===")
    
    if pose_model:
        print("Testing with basic MediaPipe...")
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_model.process(rgb_frame)
            
            if results.pose_landmarks:
                print(f"✓ Basic MediaPipe detected {len(results.pose_landmarks.landmark)} landmarks")
            else:
                print("⚠ Basic MediaPipe did not detect any landmarks")
                
        except Exception as e:
            print(f"✗ Error with basic MediaPipe processing: {e}")
    
    if custom_model:
        print("Testing with custom MediaPipeModel...")
        try:
            results = custom_model.process_frame(frame)
            keypoints = results['keypoints']
            print(f"✓ Custom model processed frame, keypoints shape: {keypoints.shape}")
            
            # Check if keypoints contain valid data
            non_zero_points = np.count_nonzero(keypoints)
            print(f"  Non-zero keypoint values: {non_zero_points}")
            
        except Exception as e:
            print(f"✗ Error with custom model processing: {e}")

def main():
    print("MediaPipe Debugging Tool")
    print("=" * 50)
    
    # Test video reading
    video_path, frame = test_video_reading()
    
    # Test MediaPipe import
    pose_model = test_mediapipe_import()
    
    # Test custom model import
    custom_model = test_custom_model_import()
    
    # Test processing if we have both video and models
    if frame is not None and (pose_model or custom_model):
        test_basic_processing(video_path, frame, pose_model, custom_model)
    
    print("\n=== Summary ===")
    if video_path:
        print(f"✓ Video can be read from: {video_path}")
    else:
        print("✗ No readable video found")
        
    if pose_model:
        print("✓ Basic MediaPipe is working")
    else:
        print("✗ Basic MediaPipe has issues")
        
    if custom_model:
        print("✓ Custom MediaPipeModel is working")
    else:
        print("✗ Custom MediaPipeModel has issues")

if __name__ == "__main__":
    main()
