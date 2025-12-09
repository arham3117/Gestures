"""
Simple Data Collection Script for Hand Gestures
Student Project - Intro to AI

Captures images from webcam for 5 hand gestures
"""

import cv2
import os
from datetime import datetime

# List of gestures we want to collect
gestures = ['fist', 'open_palm', 'pointing_finger', 'thumbs_up', 'peace_sign']

# Settings
samples_per_gesture = 200  # How many images to collect for each gesture
save_dir = 'data/raw'  # Where to save images

# Create folders for each gesture
for gesture in gestures:
    folder = os.path.join(save_dir, gesture)
    os.makedirs(folder, exist_ok=True)

# Open webcam
camera = cv2.VideoCapture(0)
print("Camera opened successfully!")
print(f"\nWe will collect {samples_per_gesture} samples for each gesture")
print("="*60)

# Collect data for each gesture
for gesture in gestures:
    print(f"\n\nGesture: {gesture.upper()}")
    print("-"*60)
    print("Instructions:")
    print("1. Position your hand in the GREEN rectangle")
    print("2. Press SPACE to start collecting")
    print("3. Press Q to move to next gesture")
    print("-"*60)

    input("\nPress ENTER when ready...")

    count = 0  # Count of collected images
    collecting = False  # Are we collecting or not?

    while count < samples_per_gesture:
        # Read frame from camera
        ret, frame = camera.read()
        if not ret:
            print("Error: Can't read from camera")
            break

        # Flip frame (mirror effect)
        frame = cv2.flip(frame, 1)

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Define where hand should be
        x1 = int(width * 0.5)
        y1 = int(height * 0.1)
        x2 = int(width * 0.9)
        y2 = int(height * 0.7)

        # Draw green rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get the hand region
        hand_region = frame[y1:y2, x1:x2]

        # Show status on screen
        status = f"{gesture.upper()} - {count}/{samples_per_gesture}"
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if collecting:
            cv2.putText(frame, "COLLECTING...", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press SPACE to start", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show frame
        cv2.imshow('Data Collection', frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # SPACE - start/stop collecting
            collecting = not collecting
            print(f"{'Started' if collecting else 'Paused'} collecting")

        elif key == ord('q'):  # Q - quit this gesture
            print(f"Stopped. Collected {count} samples")
            break

        # If collecting, save the image
        if collecting:
            # Convert to grayscale
            gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

            # Blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)

            # Resize to 200x200
            resized = cv2.resize(blurred, (200, 200))

            # Create unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{gesture}_{timestamp}.jpg"
            filepath = os.path.join(save_dir, gesture, filename)

            # Save image
            cv2.imwrite(filepath, resized)
            count += 1

            # Print progress every 10 images
            if count % 10 == 0:
                print(f"Collected {count}/{samples_per_gesture}")

            # Small delay
            cv2.waitKey(100)

    print(f"\nCompleted {gesture}: {count} samples saved!")

# Close camera
camera.release()
cv2.destroyAllWindows()

# Print summary
print("\n" + "="*60)
print("DATA COLLECTION COMPLETE!")
print("="*60)
print("\nSummary:")
for gesture in gestures:
    folder = os.path.join(save_dir, gesture)
    count = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
    print(f"  {gesture}: {count} images")
print("="*60)
