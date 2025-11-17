# Import necessary libraries
import cv2  # for camera and image processing
import mediapipe as mp  # for hand detection
import numpy as np  # for numerical operations
import os  # for file operations
import json  # for saving data
from datetime import datetime  # for timestamps


class GestureDataCollector:
    """
    This class collects hand gesture data using your webcam.
    It uses MediaPipe to detect hands and extract landmark points.
    """

    def __init__(self, output_dir='data/raw'):
        """
        Initialize the data collector.
        Sets up MediaPipe and creates folders to save data.
        """
        self.output_dir = output_dir

        # Setup MediaPipe hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # False because we're using video, not images
            max_num_hands=1,  # we only want to detect one hand
            min_detection_confidence=0.5,  # how confident the model needs to be (0.5 = 50%)
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils  # to draw landmarks on screen

        # List of gestures we want to collect
        self.gestures = ['fist', 'open_palm', 'thumbs_up', 'peace_sign', 'pointing_finger']
        self.current_gesture_idx = 0  # start with the first gesture
        self.samples_collected = {gesture: 0 for gesture in self.gestures}  # counter for each gesture

        # Create folder to save data if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def extract_landmarks(self, hand_landmarks):
        """
        Extract the 21 hand landmark points from MediaPipe.
        Normalizes them relative to the wrist position so hand size doesn't matter.
        Returns a list of 63 numbers (21 landmarks x 3 coordinates each).
        """
        landmarks = []

        # Get wrist position (landmark 0) to use as reference point
        wrist = hand_landmarks.landmark[0]

        # Go through all 21 landmarks
        for landmark in hand_landmarks.landmark:
            # Subtract wrist position from each landmark to normalize
            # This makes the model work regardless of hand size or distance from camera
            landmarks.extend([
                landmark.x - wrist.x,  # x coordinate
                landmark.y - wrist.y,  # y coordinate
                landmark.z - wrist.z   # z coordinate (depth)
            ])

        return landmarks

    def save_sample(self, landmarks, gesture_name):
        """
        Save one sample of gesture data to a JSON file.
        Each file contains the gesture name, landmark coordinates, and timestamp.
        """
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{gesture_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Organize data into a dictionary
        data = {
            'gesture': gesture_name,
            'landmarks': landmarks,
            'timestamp': timestamp
        }

        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(data, f)

        # Update counter
        self.samples_collected[gesture_name] += 1

    def run(self, samples_per_gesture=100, camera_index=0):
        """
        Main function that runs the data collection process.
        Opens webcam, shows video, and collects samples when you press SPACE.
        """
        # Open the webcam
        cap = cv2.VideoCapture(camera_index)

        # Print instructions
        print("Hand Gesture Data Collection")
        print("=" * 50)
        print("\nControls:")
        print("  SPACE - Capture sample")
        print("  N - Next gesture")
        print("  P - Previous gesture")
        print("  Q - Quit")
        print("\nGestures to collect:", ", ".join(self.gestures))
        print("=" * 50)

        # Variables to track collection state
        collecting = False  # are we currently collecting samples?
        auto_collect_count = 0  # how many samples collected so far

        # Main loop - runs until we quit
        while cap.isOpened():
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                break

            # Flip the image so it looks like a mirror
            frame = cv2.flip(frame, 1)

            # Convert from BGR to RGB (MediaPipe needs RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks using MediaPipe
            results = self.hands.process(rgb_frame)

            # Get the name of the current gesture we're collecting
            current_gesture = self.gestures[self.current_gesture_idx]

            # Get frame dimensions for text placement
            h, w, _ = frame.shape

            # Show which gesture we're collecting
            cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show progress (how many samples collected)
            cv2.putText(frame, f"Collected: {self.samples_collected[current_gesture]}/{samples_per_gesture}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Check if a hand was detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand landmarks on the screen (dots and lines)
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # If we're in collection mode, save samples
                    if collecting:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        self.save_sample(landmarks, current_gesture)
                        auto_collect_count += 1

                        # Show "COLLECTING!" message
                        cv2.putText(frame, "COLLECTING!", (w//2 - 100, h//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        # Check if we collected enough samples
                        if auto_collect_count >= samples_per_gesture:
                            collecting = False
                            auto_collect_count = 0
                            print(f"\nCompleted {current_gesture}! Moving to next gesture...")
                            # Move to next gesture
                            self.current_gesture_idx = (self.current_gesture_idx + 1) % len(self.gestures)
            else:
                # No hand detected - show warning
                cv2.putText(frame, "No hand detected!", (10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display the video frame
            cv2.imshow('Gesture Data Collection', frame)

            # Check for keyboard presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                # Quit the program
                break
            elif key == ord(' '):
                # Start/stop collecting when SPACE is pressed
                if results.multi_hand_landmarks:
                    collecting = not collecting  # toggle on/off
                    if collecting:
                        auto_collect_count = 0
                        print(f"\nStarting collection for {current_gesture}...")
            elif key == ord('n'):
                # Move to next gesture
                self.current_gesture_idx = (self.current_gesture_idx + 1) % len(self.gestures)
                collecting = False
                print(f"\nSwitched to gesture: {self.gestures[self.current_gesture_idx]}")
            elif key == ord('p'):
                # Move to previous gesture
                self.current_gesture_idx = (self.current_gesture_idx - 1) % len(self.gestures)
                collecting = False
                print(f"\nSwitched to gesture: {self.gestures[self.current_gesture_idx]}")

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        # Print final summary
        print("\n" + "=" * 50)
        print("Collection Summary:")
        for gesture, count in self.samples_collected.items():
            print(f"  {gesture}: {count} samples")
        print("=" * 50)


# This runs when you execute the script directly
if __name__ == "__main__":
    collector = GestureDataCollector()
    collector.run(samples_per_gesture=100)
