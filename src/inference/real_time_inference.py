# Import necessary libraries
import cv2  # for camera and image processing
import mediapipe as mp  # for hand detection
import numpy as np  # for numerical operations
import joblib  # for loading the trained model
import time  # for calculating FPS


class RealTimeGestureRecognizer:
    """
    This class uses the trained model to recognize gestures in real-time.
    It opens your webcam, detects hands, and predicts what gesture you're making.
    """

    def __init__(self, model_path='models/gesture_classifier.joblib'):
        """
        Initialize the recognizer.
        Loads the trained model and sets up MediaPipe for hand detection.
        """
        # Load the model we trained earlier
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

        # Setup MediaPipe hand detection (same as data collection)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # using video, not images
            max_num_hands=1,  # detect only one hand
            min_detection_confidence=0.5,  # 50% confidence to detect hand
            min_tracking_confidence=0.5  # 50% confidence to track hand
        )
        self.mp_drawing = mp.solutions.drawing_utils  # for drawing landmarks

        # Track FPS to measure performance
        self.fps_history = []

    def extract_landmarks(self, hand_landmarks):
        """
        Extract and normalize hand landmarks (same as during data collection).
        Returns 63 features that the model can understand.
        """
        landmarks = []

        # Get wrist position to use as reference
        wrist = hand_landmarks.landmark[0]

        # Go through all 21 landmarks
        for landmark in hand_landmarks.landmark:
            # Normalize relative to wrist (same as training)
            landmarks.extend([
                landmark.x - wrist.x,  # x coordinate
                landmark.y - wrist.y,  # y coordinate
                landmark.z - wrist.z   # z coordinate (depth)
            ])

        # Reshape to match what the model expects (1 row, 63 columns)
        return np.array(landmarks).reshape(1, -1)

    def predict_gesture(self, landmarks):
        """
        Use the trained model to predict what gesture this is.
        Returns the gesture name and how confident the model is (0-1).
        """
        # Get the prediction
        prediction = self.model.predict(landmarks)[0]

        # Get confidence scores for all gestures
        probabilities = self.model.predict_proba(landmarks)[0]
        # The highest score is our confidence
        confidence = np.max(probabilities)

        return prediction, confidence

    def run(self):
        """
        Main function for real-time gesture recognition.
        Opens webcam, detects hands, and shows predictions live.
        """
        # Open the webcam
        cap = cv2.VideoCapture(0)

        # Print instructions
        print("\n" + "=" * 50)
        print("Real-Time Hand Gesture Recognition")
        print("=" * 50)
        print("Controls:")
        print("  Q - Quit")
        print("=" * 50 + "\n")

        # For calculating FPS
        prev_time = time.time()

        # Main loop - runs continuously until we quit
        while cap.isOpened():
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                break

            # Flip horizontally so it looks like a mirror
            frame = cv2.flip(frame, 1)

            # Convert from BGR to RGB (MediaPipe needs RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand using MediaPipe
            results = self.hands.process(rgb_frame)

            # Get frame dimensions
            h, w, _ = frame.shape

            # Calculate frames per second (FPS) to measure performance
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)  # FPS = 1 / time per frame
            prev_time = curr_time
            self.fps_history.append(fps)

            # Show FPS on screen
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Check if a hand was detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand landmarks on screen (dots and lines)
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Extract landmarks and predict the gesture
                    landmarks = self.extract_landmarks(hand_landmarks)
                    gesture, confidence = self.predict_gesture(landmarks)

                    # Create text showing prediction and confidence
                    text = f"{gesture}: {confidence:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]

                    # Draw black rectangle behind text for better visibility
                    cv2.rectangle(frame, (10, h - 80), (text_size[0] + 20, h - 20),
                                (0, 0, 0), -1)

                    # Show gesture and confidence
                    # Green if confidence > 80%, yellow otherwise
                    color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
                    cv2.putText(frame, text, (15, h - 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            else:
                # No hand detected - show message
                cv2.putText(frame, "No hand detected", (10, h - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Display the video frame
            cv2.imshow('Hand Gesture Recognition', frame)

            # Check if user pressed 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        # Show performance summary
        avg_fps = np.mean(self.fps_history)
        print("\n" + "=" * 50)
        print("Performance Summary")
        print("=" * 50)
        print(f"Average FPS: {avg_fps:.2f}")
        print("=" * 50)


# This runs when you execute the script directly
if __name__ == "__main__":
    recognizer = RealTimeGestureRecognizer()
    recognizer.run()