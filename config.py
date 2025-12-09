"""
Configuration File - Hand Gesture Recognition
Student Project - Intro to AI

Simple settings for the project
"""

# List of gestures we recognize
GESTURES = ['fist', 'open_palm', 'pointing_finger', 'thumbs_up', 'peace_sign']

# Data Collection Settings
SAMPLES_PER_GESTURE = 200  # How many images to collect per gesture
CAPTURE_IMAGE_SIZE = (200, 200)  # Size of captured images

# Data Preprocessing Settings
MODEL_IMAGE_SIZE = (64, 64)  # Size of images for the model
TEST_SIZE = 0.2  # 20% of data for testing, 80% for training

# Model Training Settings
EPOCHS = 50  # Maximum training iterations
BATCH_SIZE = 32  # Number of images processed at once
LEARNING_RATE = 0.001  # How fast the model learns

# File Paths
RAW_DATA_DIR = 'data/raw'
PROCESSED_DATA_DIR = 'data/processed'
MODELS_DIR = 'models'
