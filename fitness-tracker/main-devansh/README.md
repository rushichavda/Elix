# Project: Pose Estimation Testing
# Pose Estimation Testing Framework for Fitness Applications

This framework provides a comprehensive testing suite for evaluating pose estimation models (MediaPipe, MoveNet) for fitness applications with real-time rep counting and form analysis.

## Features

- **Multi-Model Support**: Test and compare MediaPipe, MoveNet (Lightning & Thunder)
- **Exercise Recognition**: Automated detection for squats, shoulder press, deadlifts, and push-ups
- **Rep Counting**: Real-time repetition counting with state machine logic
- **Form Analysis**: Detect common form faults with severity ratings and corrections
- **Performance Metrics**: FPS, processing time, and accuracy measurements
- **Comprehensive Reporting**: Visual and textual reports comparing model performance

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd pose-estimation-testing
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Project Structure

```
pose-estimation-testing/
├── src/
│   ├── models/          # Pose estimation model wrappers
│   ├── analyzers/       # Rep counting and form analysis
│   ├── exercises/       # Exercise-specific logic
│   └── utils/           # Helper functions
├── scripts/             # Main execution scripts
├── data/
│   ├── videos/          # Test videos
│   └── results/         # Output results
└── notebooks/           # Jupyter notebooks for analysis
```

## Usage

### Quick Start

1. **Prepare test videos**: Place your exercise videos in `data/videos/` organized by form quality:
```
data/videos/
├── good_form/
│   ├── squat_good_1.mp4
│   └── shoulder_press_good_1.mp4
└── bad_form/
    ├── squat_knees_cave.mp4
    └── squat_forward_lean.mp4
```

2. **Run comparison on all videos**:
```bash
python scripts/run_comparison.py
```

3. **Process a single video**:
```bash
python scripts/run_comparison.py --single-video path/to/video.mp4 --exercise squat --reps 10
```

### Detailed Usage

#### Testing Different Models

```python
from src.models.mediapipe_model import MediaPipeModel
from src.models.movenet_model import MoveNetModel
from src.analyzers.rep_counter import RepCounter
from src.analyzers.form_analyzer import FormAnalyzer

# Initialize model
model = MediaPipeModel()  # or MoveNetModel('lightning')

# Process video
results = model.process_video('path/to/video.mp4', 'output.mp4')

# Analyze exercise
rep_counter = RepCounter('squat')
form_analyzer = FormAnalyzer('squat')

for i, frame_result in enumerate(results):
    keypoints = frame_result['keypoints'][0]
    
    # Count reps
    rep_result = rep_counter.process_frame(keypoints, i)
    
    # Analyze form
    faults = form_analyzer.analyze_form(keypoints, rep_result['current_state'])
```

## Form Fault Detection

The framework detects various exercise-specific faults:

### Squat Faults
- **Knee Valgus**: Knees caving inward
- **Forward Lean**: Excessive trunk forward lean
- **Insufficient Depth**: Not reaching parallel
- **Heel Raise**: Heels coming off ground
- **Knee Forward**: Knees traveling too far forward

### Shoulder Press Faults
- **Elbow Flare**: Elbows too wide
- **Wrist Bend**: Bent wrists under load
- **Back Arch**: Excessive lumbar extension
- **Asymmetry**: Uneven arm movement

### Form Scoring
- Each fault reduces the form score
- Severity levels: Minor (-5), Moderate (-10), Severe (-20)
- Real-time corrections provided

## Performance Considerations

### Model Selection Guide

| Model | FPS (Mobile) | Accuracy | Best For |
|-------|-------------|----------|----------|
| MoveNet Lightning | 30+ | Good | Real-time mobile apps |
| MoveNet Thunder | 15-20 | Better | Accuracy-focused apps |
| MediaPipe | 20-30 | Good | 3D pose + segmentation |

### Optimization Tips

1. **For Mobile Deployment**:
   - Use MoveNet Lightning
   - Process every 2-3 frames
   - Reduce input resolution
   - Use model quantization

2. **For Accuracy**:
   - Use MoveNet Thunder or MediaPipe
   - Process all frames
   - Higher input resolution
   - Post-process with smoothing

## Output Reports

The framework generates:

1. **Performance Metrics**:
   - FPS comparison charts
   - Processing time analysis
   - Resource usage stats

2. **Accuracy Analysis**:
   - Rep counting accuracy
   - Form detection rates
   - Exercise-specific performance

3. **Visual Outputs**:
   - Annotated videos with keypoints
   - Rep counting graphs
   - Form score timelines

## Common Issues & Solutions

### Low FPS on Mobile
- Switch to MoveNet Lightning
- Reduce video resolution
- Skip frames (process every 2nd or 3rd)
- Enable GPU acceleration

### Inaccurate Rep Counting
- Adjust angle thresholds in `exercise_configs`
- Increase smoothing window
- Check keypoint confidence thresholds
- Ensure proper camera angle

### Missing Form Faults
- Lower fault detection thresholds
- Add more fault rules
- Check keypoint visibility
- Improve lighting conditions

## Testing Your Own Videos

1. **Video Requirements**:
   - Clear view of full body
   - Stable camera position
   - Good lighting
   - 720p or higher resolution

2. **Naming Convention**:
   ```
   [exercise]_[quality]_[number].mp4
   Example: squat_good_1.mp4, squat_knees_cave_1.mp4
   ```

3. **Expected Output**:
   - Annotated video with keypoints
   - Rep count overlay
   - Form fault notifications
   - Summary report

## Future Enhancements

- [ ] Add more exercises (lunges, bicep curls, etc.)
- [ ] Multi-person support
- [ ] Real-time mobile app integration
- [ ] Cloud-based processing option
- [ ] Advanced biomechanics analysis
- [ ] Personalized form recommendations

## Contributing

1. Add new exercises in `src/exercises/`
2. Implement fault detection in `src/analyzers/form_analyzer.py`
3. Add model wrappers in `src/models/`
4. Submit PR with test videos and results

## License

MIT License - See LICENSE file for details