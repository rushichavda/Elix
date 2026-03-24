# Elix - Health & Fitness AI Platform

A collection of AI-powered health and fitness tools built with computer vision, machine learning, and natural language processing.

## Projects

### 1. Rep Counter (`rep-counter/`)

Real-time exercise repetition counter using computer vision and neural networks. Tracks squats and pushups with form quality assessment.

**How it works:**
- Uses MediaPipe for pose detection (body keypoints from video)
- A trained PyTorch neural network predicts exercise depth (0-1 continuous value)
- A wave pipeline algorithm detects complete reps and rates form quality (GREEN/YELLOW/RED)

**Key files:**
- `wave_pipeline.py` - Core wave pattern detection and segmentation algorithms
- `regressor_final_squats.py` - Main inference engine for squat analysis
- `pushup/` - Pushup-specific inference and models

**Tech stack:** Python, PyTorch, MediaPipe, OpenCV, Matplotlib

---

### 2. Fitness Tracker (`fitness-tracker/`)

Multi-component fitness application with AI-powered tracking and analysis.

#### Chatbot (`chatbot/`)
Streamlit-based fitness tracker with natural language exercise logging. Uses OpenAI GPT to parse inputs like "I did 3 sets of 10 pushups" and stores workout data in SQLite with analytics dashboards.

```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"
streamlit run fitness_tracker.py
```

#### Pose Estimation (`pose-estimation/`)
Framework for evaluating pose estimation models (MediaPipe, MoveNet Lightning & Thunder) on exercise form analysis. Includes rep counting, form fault detection, and performance benchmarking.

#### Exercise Classifier (`exercise-classifier/`)
ML-based exercise form classification using MediaPipe keypoints and template matching.

#### Main Devansh (`main-devansh/`)
Structured pose estimation comparison framework with modular architecture for testing different models on squat analysis.

**Tech stack:** Python, Streamlit, OpenAI, MediaPipe, MoveNet, PyTorch, Plotly, SQLite

---

## Setup

Each subproject has its own `requirements.txt`. Install dependencies per project:

```bash
cd <project-folder>
pip install -r requirements.txt
```

## Environment Variables

The fitness tracker chatbot requires:
```
OPENAI_API_KEY=your-openai-api-key
```

## License

Private repository.
