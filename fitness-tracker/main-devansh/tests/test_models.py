# Python file
# tests/test_models.py
import pytest
import numpy as np
import cv2
import time
from pathlib import Path
import sys
sys.path.append('..')

from src.models.mediapipe_model import MediaPipeModel
from src.models.movenet_model import MoveNetModel
from src.analyzers.rep_counter import RepCounter
from src.analyzers.form_analyzer import FormAnalyzer

class TestPoseModels:
    """Test suite for pose estimation models"""
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing"""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        # Add some features to make it more realistic
        cv2.circle(frame, (320, 100), 30, (0, 0, 255), -1)  # Head
        cv2.rectangle(frame, (280, 130), (360, 300), (0, 0, 0), 3)  # Body
        return frame
    
    @pytest.fixture
    def sample_keypoints(self):
        """Create sample keypoints for testing"""
        # 17 COCO keypoints with (x, y, confidence)
        keypoints = np.array([
            [320, 100, 0.9],  # nose
            [310, 90, 0.85],  # left_eye
            [330, 90, 0.85],  # right_eye
            [300, 95, 0.8],   # left_ear
            [340, 95, 0.8],   # right_ear
            [290, 150, 0.9],  # left_shoulder
            [350, 150, 0.9],  # right_shoulder
            [280, 200, 0.85], # left_elbow
            [360, 200, 0.85], # right_elbow
            [270, 250, 0.8],  # left_wrist
            [370, 250, 0.8],  # right_wrist
            [300, 300, 0.9],  # left_hip
            [340, 300, 0.9],  # right_hip
            [295, 380, 0.85], # left_knee
            [345, 380, 0.85], # right_knee
            [290, 460, 0.8],  # left_ankle
            [350, 460, 0.8]   # right_ankle
        ])
        return keypoints
    
    def test_mediapipe_initialization(self):
        """Test MediaPipe model initialization"""
        model = MediaPipeModel()
        assert model is not None
        assert model.model_name == "MediaPipe"
        assert model.model is not None
    
    def test_movenet_initialization(self):
        """Test MoveNet model initialization"""
        # Test both variants
        from typing import Literal
        for variant in ['lightning', 'thunder']:
            model = MoveNetModel(variant)  # type: ignore[arg-type]
            assert model is not None
            assert model.model_name == f"MoveNet_{variant}"
            assert model.input_size == (192 if variant == 'lightning' else 256)
    
    def test_mediapipe_inference(self, sample_frame):
        """Test MediaPipe inference"""
        model = MediaPipeModel()
        result = model.process_frame(sample_frame)
        
        assert 'keypoints' in result
        assert 'processing_time' in result
        assert result['keypoints'].shape == (1, 33, 3)  # MediaPipe has 33 keypoints
        assert result['processing_time'] > 0
    
    def test_movenet_inference(self, sample_frame):
        """Test MoveNet inference"""
        model = MoveNetModel('lightning')
        result = model.process_frame(sample_frame)
        
        assert 'keypoints' in result
        assert 'processing_time' in result
        assert result['keypoints'].shape == (1, 17, 3)  # COCO format
        assert result['processing_time'] > 0
    
    def test_performance_comparison(self, sample_frame):
        """Compare performance of different models"""
        models = {
            'mediapipe': MediaPipeModel(),
            'movenet_lightning': MoveNetModel('lightning'),
            'movenet_thunder': MoveNetModel('thunder')
        }
        
        results = {}
        num_iterations = 10
        
        for name, model in models.items():
            times = []
            for _ in range(num_iterations):
                start = time.time()
                _ = model.process_frame(sample_frame)
                times.append(time.time() - start)
            
            results[name] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'fps': 1.0 / np.mean(times)
            }
        
        # Verify performance expectations
        assert results['movenet_lightning']['fps'] > results['movenet_thunder']['fps']
        print("\nPerformance Results:")
        for name, metrics in results.items():
            print(f"{name}: {metrics['fps']:.1f} FPS")
    
    def test_rep_counter(self, sample_keypoints):
        """Test rep counter functionality"""
        rep_counter = RepCounter('squat')
        
        # Simulate squat motion
        for i in range(60):  # 60 frames
            # Modify knee position to simulate squat
            modified_kpts = sample_keypoints.copy()
            
            # Simulate down phase (frames 0-20)
            if i < 20:
                knee_offset = i * 5
                modified_kpts[13:15, 1] += knee_offset  # Move knees down
            # Simulate up phase (frames 20-40)
            elif i < 40:
                knee_offset = (40 - i) * 5
                modified_kpts[13:15, 1] += knee_offset
            
            result = rep_counter.process_frame(modified_kpts, i)
        
        assert rep_counter.rep_count >= 1
        metrics = rep_counter.get_rep_quality_metrics()
        assert 'total_reps' in metrics
        assert metrics['total_reps'] == rep_counter.rep_count
    
    def test_form_analyzer(self, sample_keypoints):
        """Test form analyzer functionality"""
        form_analyzer = FormAnalyzer('squat')
        
        # Test with good form
        faults = form_analyzer.analyze_form(sample_keypoints, 'down')
        assert isinstance(faults, list)
        
        # Test with bad form (knees caved in)
        bad_keypoints = sample_keypoints.copy()
        bad_keypoints[13, 0] -= 50  # Move left knee inward
        bad_keypoints[14, 0] += 50  # Move right knee inward
        
        faults = form_analyzer.analyze_form(bad_keypoints, 'down')
        fault_types = [f.fault_type for f in faults]
        
        # Should detect knee valgus
        assert any('knee_valgus' in ft for ft in fault_types)
        
        # Get form summary
        summary = form_analyzer.get_form_summary()
        assert 'form_score' in summary
        assert summary['form_score'] <= 100
    
    def test_model_accuracy_on_video(self):
        """Test model accuracy on a real video file"""
        video_path = Path("data/videos/good_form/squat_good_1.mp4")
        if not video_path.exists():
            pytest.skip("Test video not found")
        
        model = MediaPipeModel()
        rep_counter = RepCounter('squat')
        
        # Process video
        results = model.process_video(str(video_path))
        
        # Count reps
        for i, result in enumerate(results):
            if result['keypoints'].shape[0] > 0:
                keypoints = result['keypoints'][0]
                rep_counter.process_frame(keypoints, i)
        
        # Check if reps were detected
        assert rep_counter.rep_count > 0
        print(f"\nDetected {rep_counter.rep_count} reps in video")
    
    def test_edge_cases(self, sample_frame):
        """Test edge cases and error handling"""
        model = MediaPipeModel()
        
        # Test with empty frame
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = model.process_frame(empty_frame)
        assert result['keypoints'].shape[0] >= 0
        
        # Test with small frame
        small_frame = cv2.resize(sample_frame, (160, 120))
        result = model.process_frame(small_frame)
        assert 'keypoints' in result
        
        # Test with grayscale (should handle conversion)
        gray_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        result = model.process_frame(gray_3ch)
        assert 'keypoints' in result

@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for different scenarios"""
    
    def test_mobile_simulation(self, sample_frame):
        """Simulate mobile device constraints"""
        import time
        
        # Simulate mobile processing delays
        def add_mobile_delay():
            time.sleep(0.01)  # 10ms overhead
        
        model = MoveNetModel('lightning')
        
        # Test with frame skipping
        frames_processed = 0
        start_time = time.time()
        
        for i in range(90):  # 3 seconds at 30fps
            if i % 3 == 0:  # Process every 3rd frame
                add_mobile_delay()
                _ = model.process_frame(sample_frame)
                frames_processed += 1
        
        elapsed = time.time() - start_time
        effective_fps = frames_processed / elapsed
        
        print(f"\nMobile simulation: {effective_fps:.1f} effective FPS")
        assert effective_fps >= 10  # Minimum acceptable for mobile
    
    def test_batch_processing(self, sample_frame):
        """Test batch processing capabilities"""
        model = MoveNetModel('lightning')
        
        # Create batch of frames
        batch_size = 4
        batch_frames = np.stack([sample_frame] * batch_size)
        
        # Test batch processing if supported
        if hasattr(model, 'process_frame_batch'):
            result = model.process_frame_batch(batch_frames)
            assert result['keypoints'].shape[0] == batch_size
            print(f"\nBatch processing time: {result['batch_processing_time']:.3f}s")

def run_all_tests():
    """Run all tests and generate report"""
    pytest.main([__file__, '-v', '--tb=short'])

if __name__ == "__main__":
    run_all_tests()