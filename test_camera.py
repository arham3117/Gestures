import cv2

print("Testing camera detection...")
print("=" * 50)

# Try to detect available cameras
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w, c = frame.shape
            print(f"Camera {i}: Detected! Resolution: {w}x{h}")
            cap.release()
        else:
            print(f"Camera {i}: Found but cannot read frames")
            cap.release()
    else:
        if i == 0:
            print(f"Camera {i}: Not found")

print("=" * 50)
print("\nIf you see 'Camera 0: Detected!' above, Camo is working!")
print("If not, try:")
print("1. Make sure Camo app is running on your phone")
print("2. Check Camo is connected to your Mac")
print("3. Grant camera permissions if prompted")
