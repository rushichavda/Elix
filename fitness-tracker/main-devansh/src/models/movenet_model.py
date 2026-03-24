import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import time
from typing import Dict, Literal
from .base_model import BasePoseModel


class MoveNetModel(BasePoseModel):
    """MoveNet (TensorFlow) pose estimation model wrapper"""

    MODELS = {
        'lightning': 'https://tfhub.dev/google/movenet/singlepose/lightning/4',
        'thunder': 'https://tfhub.dev/google/movenet/singlepose/thunder/4'
    }

    INPUT_SIZE = {
        'lightning': 192,
        'thunder': 256
    }

    def __init__(self, model_type: Literal['lightning', 'thunder'] = 'lightning'):
        super().__init__(f"MoveNet_{model_type}")
        self.model_type = model_type
        self.input_size = self.INPUT_SIZE[model_type]
        self.load_model()

    def load_model(self):
        """Load MoveNet model from TensorFlow Hub"""
        print(f"Loading MoveNet {self.model_type} model...")
        self.model = hub.load(self.MODELS[self.model_type])
        self.movenet = self.model.signatures['serving_default']

        # Warm up the model with correct dtype and keyword argument
        dummy_input = tf.zeros((1, self.input_size, self.input_size, 3), dtype=tf.int32)
        _ = self.movenet(input=dummy_input)
        print(f"MoveNet {self.model_type} loaded successfully!")

    def preprocess_frame(self, frame: np.ndarray) -> tuple:
        """Preprocess frame for MoveNet input"""
        # Resize and pad to square
        height, width = frame.shape[:2]

        if height > width:
            scale = self.input_size / height
            new_height = self.input_size
            new_width = int(width * scale)
        else:
            scale = self.input_size / width
            new_width = self.input_size
            new_height = int(height * scale)

        resized = cv2.resize(frame, (new_width, new_height))

        # Pad to square
        pad_top = (self.input_size - new_height) // 2
        pad_bottom = self.input_size - new_height - pad_top
        pad_left = (self.input_size - new_width) // 2
        pad_right = self.input_size - new_width - pad_left

        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # Convert to tensor with correct dtype
        input_tensor = tf.convert_to_tensor(padded[np.newaxis, ...], dtype=tf.int32)

        return input_tensor, scale, (pad_left, pad_top)

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process frame with MoveNet"""
        start_time = time.time()

        # Preprocess
        input_tensor, scale, (pad_x, pad_y) = self.preprocess_frame(frame)

        # Run inference with keyword argument
        outputs = self.movenet(input=input_tensor)
        keypoints_with_scores = outputs['output_0'].numpy()

        # Convert keypoints to original image coordinates
        keypoints = np.zeros((1, 17, 3))
        for i in range(17):
            y, x, score = keypoints_with_scores[0, 0, i]
            # Convert from normalized padded coordinates to original image coordinates
            orig_x = (x * self.input_size - pad_x) / scale
            orig_y = (y * self.input_size - pad_y) / scale
            keypoints[0, i] = [orig_x, orig_y, score]

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        return {
            'keypoints': keypoints,
            'processing_time': processing_time,
            'frame_dimensions': (frame.shape[0], frame.shape[1]),
            'raw_output': keypoints_with_scores
        }

    def process_frame_batch(self, frames: np.ndarray) -> Dict:
        """Process multiple frames in batch for better efficiency"""
        start_time = time.time()

        batch_size = frames.shape[0]
        batch_inputs = []
        scales = []
        paddings = []

        # Preprocess all frames
        for frame in frames:
            input_tensor, scale, padding = self.preprocess_frame(frame)
            batch_inputs.append(input_tensor[0])  # Remove batch dimension
            scales.append(scale)
            paddings.append(padding)

        # Stack into batch and ensure correct dtype
        batch_tensor = tf.stack(batch_inputs)
        batch_tensor = tf.cast(batch_tensor, tf.int32)  # Ensure int32 dtype

        # Run batch inference with keyword argument
        outputs = self.movenet(input=batch_tensor)
        keypoints_batch = outputs['output_0'].numpy()

        # Convert all keypoints
        all_keypoints = np.zeros((batch_size, 1, 17, 3))
        for b in range(batch_size):
            for i in range(17):
                y, x, score = keypoints_batch[b, 0, i]
                orig_x = (x * self.input_size - paddings[b][0]) / scales[b]
                orig_y = (y * self.input_size - paddings[b][1]) / scales[b]
                all_keypoints[b, 0, i] = [orig_x, orig_y, score]

        processing_time = time.time() - start_time
        avg_time_per_frame = processing_time / batch_size

        return {
            'keypoints': all_keypoints,
            'processing_time': avg_time_per_frame,
            'batch_processing_time': processing_time,
            'batch_size': batch_size
        }

    def get_model_details(self) -> Dict:
        """Get model-specific details"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'input_size': self.input_size,
            'expected_fps': 30 if self.model_type == 'lightning' else 15,
            'keypoint_count': 17,
            'supports_batch': True
        }