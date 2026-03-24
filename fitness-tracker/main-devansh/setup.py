# Python file
# setup.py
import os
import sys
import subprocess
import platform
from pathlib import Path

class SetupPoseEstimation:
    """Setup script for pose estimation testing framework"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.python_version = sys.version_info
        self.os_type = platform.system()
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        print(f"Python version: {self.python_version.major}.{self.python_version.minor}")
        
        if self.python_version < (3, 7):
            print("ERROR: Python 3.7 or higher is required")
            return False
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            'data/videos/good_form',
            'data/videos/bad_form',
            'data/results',
            'logs'
        ]
        
        for dir_path in directories:
            full_path = self.root_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("\nInstalling Python dependencies...")
        
        # Upgrade pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        requirements_file = self.root_dir / "requirements.txt"
        if requirements_file.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
        else:
            # Install core dependencies manually
            core_deps = [
                "numpy>=1.21.0",
                "opencv-python>=4.5.0",
                "tensorflow>=2.8.0",
                "tensorflow-hub>=0.12.0",
                "mediapipe>=0.9.0",
                "pandas>=1.3.0",
                "matplotlib>=3.4.0",
                "tqdm>=4.62.0"
            ]
            
            for dep in core_deps:
                print(f"Installing {dep}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    
    def setup_gpu_support(self):
        """Setup GPU support if available"""
        print("\nChecking GPU support...")
        
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                print(f"Found {len(gpus)} GPU(s):")
                for gpu in gpus:
                    print(f"  - {gpu.name}")
                
                # Try to install GPU dependencies
                if self.os_type == "Linux":
                    print("\nFor GPU support, ensure you have:")
                    print("  - NVIDIA drivers (>= 450.x)")
                    print("  - CUDA 11.2")
                    print("  - cuDNN 8.1")
            else:
                print("No GPU found. Will use CPU for inference.")
                
        except Exception as e:
            print(f"Could not check GPU support: {e}")
    
    def download_test_data(self):
        """Download or create test data"""
        print("\nSetting up test data...")
        
        # Create synthetic test videos
        try:
            from scripts.download_test_videos import VideoDownloader
            downloader = VideoDownloader()
            downloader.create_sample_videos()
            print("Created synthetic test videos")
        except Exception as e:
            print(f"Could not create test videos: {e}")
            print("You'll need to add your own test videos to data/videos/")
    
    def verify_installation(self):
        """Verify that everything is installed correctly"""
        print("\nVerifying installation...")
        
        # Test imports
        modules_to_test = [
            ("TensorFlow", "tensorflow"),
            ("MediaPipe", "mediapipe"),
            ("OpenCV", "cv2"),
            ("NumPy", "numpy"),
            ("Pandas", "pandas")
        ]
        
        all_good = True
        for name, module in modules_to_test:
            try:
                __import__(module)
                print(f"✓ {name} installed correctly")
            except ImportError:
                print(f"✗ {name} not installed")
                all_good = False
        
        # Test model loading
        print("\nTesting model loading...")
        try:
            from src.models.mediapipe_model import MediaPipeModel
            model = MediaPipeModel()
            print("✓ MediaPipe model loaded successfully")
        except Exception as e:
            print(f"✗ MediaPipe model failed to load: {e}")
            all_good = False
        
        try:
            from src.models.movenet_model import MoveNetModel
            model = MoveNetModel('lightning')
            print("✓ MoveNet model loaded successfully")
        except Exception as e:
            print(f"✗ MoveNet model failed to load: {e}")
            all_good = False
        
        return all_good
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "="*50)
        print("Setup Complete! Next Steps:")
        print("="*50)
        print("\n1. Test the installation:")
        print("   python scripts/run_comparison.py --single-video data/videos/good_form/synthetic_squat_good_form.mp4 --exercise squat")
        print("\n2. Run the full comparison:")
        print("   python scripts/run_comparison.py")
        print("\n3. Open the Jupyter notebook for interactive testing:")
        print("   jupyter notebook notebooks/model_comparison_demo.ipynb")
        print("\n4. Run the test suite:")
        print("   pytest tests/test_models.py -v")
        print("\n5. Check the results in:")
        print("   data/results/")
        print("\nFor more information, see README.md")
    
    def run(self):
        """Run the complete setup process"""
        print("Pose Estimation Testing Framework Setup")
        print("=" * 50)
        
        # Check Python version
        if not self.check_python_version():
            return
        
        # Create directories
        print("\nCreating directory structure...")
        self.create_directories()
        
        # Install dependencies
        self.install_dependencies()
        
        # Setup GPU support
        self.setup_gpu_support()
        
        # Download test data
        self.download_test_data()
        
        # Verify installation
        if self.verify_installation():
            print("\n✅ Installation successful!")
        else:
            print("\n⚠️  Some components failed to install")
            print("Check the error messages above and try:")
            print("  - Updating pip: python -m pip install --upgrade pip")
            print("  - Installing missing dependencies manually")
        
        # Next steps
        self.print_next_steps()

def main():
    """Main entry point"""
    setup = SetupPoseEstimation()
    
    try:
        setup.run()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
    except Exception as e:
        print(f"\n\nError during setup: {e}")
        print("Please check the error message and try again")

if __name__ == "__main__":
    main()