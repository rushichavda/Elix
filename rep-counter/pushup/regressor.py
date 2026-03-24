import cv2
import mediapipe as mp
import torch
import torch.nn as nn

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

counter = 0
current_state = None  # 0 = Up, 1 = Down
current_leg = "right"

class SquatRegressor(nn.Module):
    def __init__(self, input_size=4):
        super(SquatRegressor, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        return torch.sigmoid(self.output(x))

def normalize_keypoints(keypoints):
    # keypoints: [ankle_x, ankle_y, knee_x, knee_y, hip_x, hip_y]
    knee_x, knee_y = keypoints[2], keypoints[3]
    return [
        keypoints[0] - knee_x,  # Ankle x relative to knee
        keypoints[1] - knee_y,  # Ankle y relative to knee
        keypoints[4] - knee_x,  # Hip x relative to knee
        keypoints[5] - knee_y   # Hip y relative to knee
    ]

def draw_best_leg(frame, results):
    global current_leg
    if not results.pose_landmarks:
        return

    landmarks = results.pose_landmarks.landmark
    RIGHT = [12, 14, 16]
    LEFT = [11, 13, 15]
    selected = RIGHT if current_leg == "right" else LEFT
    connections = [(12, 14), (14, 16)] if current_leg == "right" else [(11, 13), (13, 15)]

    if not all(landmarks[i].visibility > 0.5 for i in selected):
        return

    for i, lm in enumerate(landmarks):
        if i not in selected:
            lm.visibility = 0.0

    mp_drawing.draw_landmarks(
        frame,
        results.pose_landmarks,
        connections,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
    )

def extract_keypoints(results):
    global current_leg
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    RIGHT = [12, 14, 16]  # shoulder, elbow, wrist
    LEFT = [11, 13, 15]

    vis_right = [landmarks[i].visibility for i in RIGHT]
    vis_left = [landmarks[i].visibility for i in LEFT]

    right_valid = all(v > 0.5 for v in vis_right)
    left_valid = all(v > 0.5 for v in vis_left)

    if not right_valid and not left_valid:
        return None

    right_sum = sum(vis_right)
    left_sum = sum(vis_left)

    if current_leg == "right" and left_valid and (left_sum - right_sum) > 0.05:
        current_leg = "left"
    elif current_leg == "left" and right_valid and (right_sum - left_sum) > 0.05:
        current_leg = "right"

    selected = RIGHT if current_leg == "right" else LEFT
    keypoints = [
        landmarks[selected[2]].x, landmarks[selected[2]].y,  # Ankle
        landmarks[selected[1]].x, landmarks[selected[1]].y,  # Knee
        landmarks[selected[0]].x, landmarks[selected[0]].y   # Hip
    ]
    return normalize_keypoints(keypoints)

def classify_state(model, keypoints):
    model.eval()
    with torch.no_grad():
        data = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0)
        output = model(data)
        value = output.item()
        update_counter(value)
        return value

def update_counter(value):
    global counter, current_state
    if value > 0.8:
        current_state = 1  # Down
    elif value < 0.2:
        if current_state == 1:
            counter += 1
        current_state = 0  # Up
        
def inference(video_path, model_path="pushups_dataset_sigmoid.pkl"):
    global counter, current_state
    counter = 0
    current_state = None

    model = SquatRegressor()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # ← Use webcam

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Starting live webcam inference. Press 'q' or 'ESC' to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            keypoints = extract_keypoints(results)
            draw_best_leg(frame, results)
            value = -1
            if keypoints:
                value = classify_state(model, keypoints)
            cv2.putText(frame, f"Val: {value:.2f}", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
            cv2.putText(frame, f"Reps: {counter}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)

            cv2.imshow('Squat Counter - Live', frame)

            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    video_path = "The Perfect Push Up _ Do it right!.mp4"
    inference(video_path)