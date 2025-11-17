import cv2
import sys

print("Camera Selector for Camo")
print("=" * 50)
print("This will help you find the right camera index.")
print("Press 'q' to try the next camera, or 'c' to use this camera for data collection")
print("=" * 50)

camera_index = 0

while True:
    print(f"\nTrying camera index: {camera_index}")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Camera {camera_index} not available.")
        camera_index += 1
        if camera_index > 10:
            print("No more cameras found.")
            break
        continue

    print(f"Camera {camera_index} opened successfully!")
    print("Press 'q' to try next camera, 'c' to confirm this camera, or ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read from this camera")
            break

        # Display camera info
        h, w, _ = frame.shape
        cv2.putText(frame, f"Camera Index: {camera_index}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {w}x{h}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' for next camera, 'c' to confirm", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Camera Selection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Trying next camera...")
            break
        elif key == ord('c'):
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n{'=' * 50}")
            print(f"Selected Camera Index: {camera_index}")
            print(f"{'=' * 50}")
            print(f"\nTo use this camera for data collection, run:")
            print(f"python -c \"from src.data_collection.collect_gestures import GestureDataCollector; collector = GestureDataCollector(); collector.run(samples_per_gesture=100, camera_index={camera_index})\"")
            sys.exit(0)
        elif key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    camera_index += 1

    if camera_index > 10:
        print("Checked all camera indices up to 10.")
        break

cv2.destroyAllWindows()