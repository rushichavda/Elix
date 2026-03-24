import os
import sys
from pathlib import Path
import yt_dlp
import cv2

class VideoDownloader:
    """Download and prepare test videos from YouTube"""
    
    def __init__(self, output_dir: str = "data/videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Example video URLs (replace with actual URLs)
        self.test_videos = {
            'good_form': {
                'squat_good_1': {
                    'url': 'https://youtube.com/watch?v=EXAMPLE1',
                    'start_time': 10,  # seconds
                    'duration': 30,
                    'description': 'Perfect squat form demonstration'
                },
                'shoulder_press_good_1': {
                    'url': 'https://youtube.com/watch?v=EXAMPLE2',
                    'start_time': 15,
                    'duration': 25,
                    'description': 'Proper overhead press technique'
                }
            },
            'bad_form': {
                'squat_knees_cave': {
                    'url': 'https://youtube.com/watch?v=EXAMPLE3',
                    'start_time': 5,
                    'duration': 20,
                    'description': 'Common mistake - knee valgus'
                },
                'squat_forward_lean': {
                    'url': 'https://youtube.com/watch?v=EXAMPLE4',
                    'start_time': 8,
                    'duration': 15,
                    'description': 'Excessive forward lean demonstration'
                }
            }
        }
    
    def download_video(self, url: str, output_path: str):
        """Download video from YouTube"""
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit to 720p for reasonable file size
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"Downloading: {url}")
                ydl.download([url])
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def extract_clip(self, input_path: str, output_path: str, 
                     start_time: int, duration: int):
        """Extract specific clip from video"""
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Skip to start time
        start_frame = start_time * fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        frames_to_extract = duration * fps
        for _ in range(frames_to_extract):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        print(f"Extracted clip: {output_path}")
    
    def prepare_test_videos(self):
        """Download and prepare all test videos"""
        print("Preparing test videos...")
        
        for category, videos in self.test_videos.items():
            category_dir = self.output_dir / category
            category_dir.mkdir(exist_ok=True)
            
            for name, info in videos.items():
                output_file = category_dir / f"{name}.mp4"
                
                if output_file.exists():
                    print(f"Skipping {name} - already exists")
                    continue
                
                # Download full video
                temp_file = category_dir / f"{name}_temp.mp4"
                if self.download_video(info['url'], str(temp_file)):
                    # Extract clip
                    self.extract_clip(
                        str(temp_file),
                        str(output_file),
                        info['start_time'],
                        info['duration']
                    )
                    # Remove temp file
                    temp_file.unlink()
    
    def create_sample_videos(self):
        """Create synthetic test videos for testing"""
        print("Creating synthetic test videos...")
        
        # Create simple test patterns
        for category in ['good_form', 'bad_form']:
            category_dir = self.output_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Create a simple video with moving circle (simulating body movement)
            output_path = category_dir / f"synthetic_squat_{category}.mp4"
            self._create_synthetic_squat_video(output_path, category == 'good_form')
    
    def _create_synthetic_squat_video(self, output_path: Path, good_form: bool):
        """Create synthetic squat video for testing"""
        width, height = 640, 480
        fps = 30
        duration = 10  # seconds
        
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame_idx in range(fps * duration):
            # Create blank frame
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Simulate squat motion
            t = frame_idx / fps
            squat_phase = (np.sin(t * 2 * np.pi / 3) + 1) / 2  # 3-second cycle
            
            # Draw "skeleton"
            # Head
            head_y = int(100 + squat_phase * 100)
            cv2.circle(frame, (width//2, head_y), 20, (0, 0, 255), -1)
            
            # Body
            body_bottom = head_y + 150
            cv2.line(frame, (width//2, head_y + 20), 
                    (width//2, body_bottom), (0, 0, 0), 3)
            
            # Arms
            cv2.line(frame, (width//2 - 50, head_y + 50),
                    (width//2 + 50, head_y + 50), (0, 0, 0), 3)
            
            # Legs
            knee_y = body_bottom + int(50 + squat_phase * 50)
            ankle_y = knee_y + 80
            
            if good_form:
                # Good form - knees track properly
                knee_x_offset = 30
            else:
                # Bad form - knees cave in during squat
                knee_x_offset = int(30 - squat_phase * 20)
            
            # Left leg
            cv2.line(frame, (width//2 - 20, body_bottom),
                    (width//2 - knee_x_offset, knee_y), (0, 0, 0), 3)
            cv2.line(frame, (width//2 - knee_x_offset, knee_y),
                    (width//2 - 30, ankle_y), (0, 0, 0), 3)
            
            # Right leg
            cv2.line(frame, (width//2 + 20, body_bottom),
                    (width//2 + knee_x_offset, knee_y), (0, 0, 0), 3)
            cv2.line(frame, (width//2 + knee_x_offset, knee_y),
                    (width//2 + 30, ankle_y), (0, 0, 0), 3)
            
            # Add frame number
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            out.write(frame)
        
        out.release()
        print(f"Created synthetic video: {output_path}")

def main():
    downloader = VideoDownloader()
    
    # Create synthetic videos for immediate testing
    downloader.create_sample_videos()
    
    # Uncomment to download real videos (requires valid YouTube URLs)
    # downloader.prepare_test_videos()
    
    print("\nTest videos prepared!")
    print(f"Location: {downloader.output_dir}")

if __name__ == "__main__":
    # Add yt-dlp to requirements if downloading from YouTube
    # pip install yt-dlp
    main()