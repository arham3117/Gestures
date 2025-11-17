"""
Configuration file for GestureFlow project
"""

# Data collection settings
DATA_COLLECTION = {
    'raw_data_dir': 'data/raw',
    'samples_per_gesture': 100,
    'gestures': ['fist', 'open_palm', 'thumbs_up', 'peace_sign', 'pointing_finger']
}

# MediaPipe settings
MEDIAPIPE = {
    'max_num_hands': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Data preprocessing settings
PREPROCESSING = {
    'processed_data_dir': 'data/processed',
    'test_size': 0.2,
    'random_state': 42
}

# Model training settings
TRAINING = {
    'model_dir': 'models',
    'n_estimators': 100,
    'max_depth': 20,
    'random_state': 42,
    'model_filename': 'gesture_classifier.joblib'
}

# Inference settings
INFERENCE = {
    'confidence_threshold': 0.8,
    'camera_index': 0
}

# Performance targets
PERFORMANCE = {
    'target_accuracy': 0.85,
    'min_fps': 20
}