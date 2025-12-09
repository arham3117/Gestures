"""
Real-Time Hand Gesture Recognition
Student Project - Intro to AI
"""

import cv2
import numpy as np
from tensorflow import keras
import pickle

# Load the trained model
print("Loading model...")
model = keras.models.load_model('../models/gesture_model.h5')

# Load gesture names
with open('../data/processed/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
gestures = list(label_encoder.classes_)

print(f"Gestures: {', '.join(gestures)}")
print("\nStarting camera...")
print("Position your hand in the green rectangle")
print("Press 'q' to quit\n")

# Open webcam
camera = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = camera.read()
    if not ret:
        break

    # Flip frame so it acts like a mirror
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Define region of interest (ROI) - where hand should be
    x1, y1 = int(width * 0.5), int(height * 0.1)
    x2, y2 = int(width * 0.9), int(height * 0.7)

    # Draw green rectangle showing where to put hand
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Extract the hand region
    hand_region = frame[y1:y2, x1:x2]

    # Preprocess the hand region (same as training)
    # 1. Convert to grayscale
    gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

    # 2. Blur to reduce noise
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Resize to 64x64 (model input size)
    resized = cv2.resize(gray, (64, 64))

    # 4. Normalize pixel values to 0-1
    normalized = resized / 255.0

    # 5. Reshape for model (add batch and channel dimensions)
    input_image = normalized.reshape(1, 64, 64, 1)

    # Get prediction from model
    prediction = model.predict(input_image, verbose=0)[0]

    # Get the gesture with highest probability
    gesture_index = np.argmax(prediction)
    gesture_name = gestures[gesture_index]
    confidence = prediction[gesture_index]

    # Draw prediction on screen
    # Background box for text (made taller to fit all 5 gestures)
    cv2.rectangle(frame, (10, 10), (320, 240), (0, 0, 0), -1)

    # Title
    cv2.putText(frame, "Gesture Recognition", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Predicted gesture
    color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
    cv2.putText(frame, f"Gesture: {gesture_name.upper()}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Confidence
    cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show all probabilities
    y = 130
    for i, prob in enumerate(prediction):
        gesture = gestures[i]
        # Draw probability bar
        bar_length = int(prob * 150)
        cv2.rectangle(frame, (130, y-10), (130 + bar_length, y+5),
                     (0, 255, 0), -1)
        # Show percentage
        cv2.putText(frame, f"{gesture}: {prob:.0%}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 22

    # Show instructions
    cv2.putText(frame, "Press 'q' to quit", (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
camera.release()
cv2.destroyAllWindows()
print("Goodbye!")
