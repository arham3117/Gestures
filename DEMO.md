# Live Demo Instructions

Since this is a webcam-based application, you need to run it locally on your computer to try it out. Here's how:

## Quick Start (Try the Pre-trained Model)

If you just want to see it work without training your own model:

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/GestureFlow.git
cd GestureFlow
```

2. **Set up the environment:**
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Run the demo:**
```bash
python src/inference/real_time_inference.py
```

4. **Make the gestures:**
- Fist
- Open Palm
- Thumbs Up
- Peace Sign
- Pointing Finger

5. **Press Q to quit**

## Demo Video

[Link to demo video will go here]

## Screenshots

[Screenshots of the application will go here]

## What to Expect

- The webcam window opens
- Green dots and lines appear on your hand (these are the 21 landmarks)
- At the bottom, you'll see the predicted gesture and confidence score
- The text turns green when confidence is above 80%
- FPS is displayed at the top

## Tips for Best Results

- Good lighting helps a lot
- Keep your hand in the camera frame
- Make clear, distinct gestures
- Try different angles - the model is pretty robust!

## System Requirements

- **Python**: 3.10, 3.11, or 3.12 (NOT 3.13)
- **Webcam**: Any webcam or phone camera via apps like Camo
- **OS**: Works on Mac, Windows, and Linux
- **RAM**: 4GB minimum

## Troubleshooting

**"Model not found" error:**
- You need to train the model first, or make sure `models/gesture_classifier.joblib` exists

**Camera not working:**
- Make sure no other app is using the camera
- Check camera permissions in your system settings

**Import errors:**
- Make sure you activated the virtual environment
- Run `pip install -r requirements.txt` again

## Want to Train Your Own Model?

Follow the full instructions in the main [README.md](README.md)
