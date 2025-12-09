# Hand Gesture Recognition System

**Real-time hand gesture recognition using OpenCV and Convolutional Neural Networks (CNN)**

![Accuracy](https://img.shields.io/badge/Accuracy-99.83%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange)

---

## 🎯 Project Overview

This project implements a real-time hand gesture recognition system that classifies 5 different hand gestures with **99.83% accuracy**. The system uses computer vision (OpenCV) for image capture and preprocessing, and a Convolutional Neural Network (CNN) for classification.

### Recognized Gestures:
1. **Fist** - Closed hand with fingers curled
2. **Open Palm** - Hand with all fingers extended and spread
3. **Pointing Finger** - Index finger extended, other fingers closed
4. **Thumbs Up** - Thumb extended upward, other fingers closed
5. **Peace Sign** - Index and middle fingers extended in V shape

---

## 🚀 Key Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 99.83% |
| **Dataset Size** | 2,000 → 6,000 (after augmentation) |
| **Training Set** | 4,800 images (80%) |
| **Test Set** | 1,200 images (20%) |
| **Errors** | Only 2 out of 1,200 test samples |
| **Real-time FPS** | ~30 FPS |
| **Model Size** | 15 MB |

---

## 📁 Project Structure

```
GestureFlow/
├── data/
│   ├── raw/                    # Raw collected images (2,000 images)
│   │   ├── fist/              # 400 fist gesture images
│   │   ├── open_palm/         # 400 open palm images
│   │   ├── pointing_finger/   # 400 pointing finger images
│   │   ├── thumbs_up/         # 400 thumbs up images
│   │   └── peace_sign/        # 400 peace sign images
│   │
│   └── processed/             # Preprocessed data (6,000 augmented)
│       ├── X_train.npy        # Training images (4,800)
│       ├── X_test.npy         # Test images (1,200)
│       ├── y_train.npy        # Training labels
│       ├── y_test.npy         # Test labels
│       ├── label_encoder.pkl  # Label encoder for predictions
│       ├── sample_images.png  # Sample visualization
│       └── class_distribution.png  # Dataset balance graph
│
├── models/                    # Trained models and results
│   ├── gesture_model.h5       # Trained CNN model (15 MB)
│   ├── training_history.png   # Training/validation curves
│   ├── confusion_matrix.png   # Accuracy visualization
│   └── training_summary.txt   # Training metrics
│
├── src/                       # Source code
│   ├── collect_data.py        # Data collection script (Python)
│   ├── preprocess_data.ipynb  # Preprocessing notebook (Jupyter)
│   ├── train_model.ipynb      # Model training notebook (Jupyter)
│   └── predict_live.py        # Real-time prediction (Python)
│
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.8 or higher
- Webcam
- pip (Python package manager)

### 2. Clone Repository
```bash
git clone https://github.com/yourusername/GestureFlow.git
cd GestureFlow
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Packages:**
- `opencv-python` - Computer vision and image processing
- `tensorflow` - Deep learning framework
- `numpy` - Numerical operations
- `scikit-learn` - Data preprocessing and metrics
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization
- `jupyter` - Jupyter notebook support

---

## 📊 Complete Workflow

### Step 1: Data Collection

**Run the data collection script:**
```bash
python src/collect_data.py
```

**Instructions:**
1. Position your hand in the **green rectangle** shown on screen
2. Press **SPACE** to start/pause collecting samples
3. Press **Q** to move to the next gesture
4. Collect 200+ samples per gesture for best results

**Tips for Quality Data:**
- Ensure good, consistent lighting
- Keep hand clearly visible in the Region of Interest (ROI)
- Vary hand positions and angles slightly
- Maintain consistent distance from camera
- Use plain background for better detection

**Output:** 2,000 raw images (400 per gesture) saved in `data/raw/`

---

### Step 2: Data Preprocessing

**Open and run the preprocessing notebook:**
```bash
jupyter notebook src/preprocess_data.ipynb
```

**What this does:**
1. Loads all collected images from `data/raw/`
2. Applies data augmentation (3x expansion):
   - **Horizontal Flip** - Simulates left/right hand
   - **Rotation (±15°)** - Handles angle variations
   - **Brightness Adjustment** - Adapts to lighting changes
3. Normalizes pixel values to [0, 1] range
4. Encodes labels (text → numbers)
5. Splits into 80% training, 20% testing (stratified)
6. Saves processed data as `.npy` files

**Results:**
- Original: 2,000 images
- After augmentation: 6,000 images
- Training set: 4,800 images
- Test set: 1,200 images

**Output Files:**
- `data/processed/X_train.npy` - Training images
- `data/processed/X_test.npy` - Test images
- `data/processed/y_train.npy` - Training labels
- `data/processed/y_test.npy` - Test labels
- `data/processed/label_encoder.pkl` - For predictions
- `data/processed/sample_images.png` - Visualization
- `data/processed/class_distribution.png` - Dataset balance

---

### Step 3: Model Training

**Open and run the training notebook:**
```bash
jupyter notebook src/train_model.ipynb
```

**CNN Architecture:**
```
Input (64×64×1 grayscale image)
    ↓
Convolutional Block 1
  - Conv2D (32 filters, 3×3)
  - MaxPooling (2×2)
  - Dropout (25%)
    ↓
Convolutional Block 2
  - Conv2D (64 filters, 3×3)
  - MaxPooling (2×2)
  - Dropout (25%)
    ↓
Convolutional Block 3
  - Conv2D (128 filters, 3×3)
  - MaxPooling (2×2)
  - Dropout (25%)
    ↓
Fully Connected Layers
  - Flatten
  - Dense (256 units, ReLU)
  - Dropout (50%)
  - Dense (128 units, ReLU)
  - Dropout (50%)
    ↓
Output Layer
  - Dense (5 units, Softmax)
```

**Training Configuration:**
- **Optimizer:** Adam
- **Learning Rate:** 0.001 (adaptive reduction)
- **Batch Size:** 32
- **Epochs:** 50 (with early stopping)
- **Loss Function:** Categorical Cross-Entropy

**Training Duration:** ~5 minutes on CPU

**Output:**
- `models/gesture_model.h5` - Trained model
- `models/training_history.png` - Training curves
- `models/training_summary.txt` - Metrics

---

### Step 4: Real-Time Prediction

**Run the real-time prediction script:**
```bash
python src/predict_live.py
```

**How it works:**
1. Loads the trained model
2. Opens webcam feed
3. Extracts hand region from ROI
4. Preprocesses image (grayscale, blur, resize, normalize)
5. Predicts gesture using CNN
6. Displays result with confidence score

**Controls:**
- Position hand in **green rectangle**
- Press **'q'** to quit

**Performance:**
- Real-time: ~30 FPS
- Inference time: ~30-50ms per frame
- Smooth, responsive predictions

---

## 📈 Model Performance

### Overall Accuracy: 99.83%

### Per-Class Performance:

| Gesture | Precision | Recall | F1-Score | Accuracy | Test Samples |
|---------|-----------|--------|----------|----------|--------------|
| Fist | 100.0% | 100.0% | 100.0% | 100.0% | 240 |
| Open Palm | 100.0% | 100.0% | 100.0% | 100.0% | 240 |
| Peace Sign | 100.0% | 99.2% | 99.6% | 99.2% | 240 |
| Pointing Finger | 99.2% | 100.0% | 99.6% | 100.0% | 240 |
| Thumbs Up | 100.0% | 100.0% | 100.0% | 100.0% | 240 |
| **Average** | **99.8%** | **99.8%** | **99.8%** | **99.83%** | **1,200** |

### Confusion Matrix:
Only 2 misclassifications out of 1,200 test samples:
- 2 Peace Sign samples predicted as Pointing Finger
- All other predictions: 100% accurate

---

## 🧠 Technical Details

### Data Preprocessing
- **Image Size:** 64×64 pixels (grayscale)
- **Augmentation:** 3× expansion (flips, rotations, brightness)
- **Normalization:** Pixel values scaled to [0, 1]
- **Split Ratio:** 80% train, 20% test (stratified)

### Model Architecture
- **Type:** Convolutional Neural Network (CNN)
- **Layers:** 3 conv blocks + 2 dense layers
- **Parameters:** ~3.9M total, ~1.3M trainable
- **Regularization:** Dropout (25% conv, 50% dense)
- **Activation:** ReLU (hidden), Softmax (output)

### Why CNN?
1. **Spatial Feature Learning:** Automatically learns patterns from images
2. **Translation Invariance:** Robust to hand position variations
3. **Parameter Efficiency:** Shared weights reduce complexity
4. **Proven Performance:** State-of-the-art for image classification
5. **Real-time Capable:** Fast inference for live predictions

### Training Features
- **Early Stopping:** Stops when validation loss stops improving (patience: 10)
- **Learning Rate Reduction:** Adaptive LR reduction (factor: 0.5, patience: 5)
- **Model Checkpoint:** Saves best model based on validation accuracy

---

## 💡 Why This Project?

### Applications:
- **Touchless Control Systems** - Post-pandemic relevance
- **Accessibility Tools** - For motor-impaired users
- **Human-Robot Interaction** - Gesture-based commands
- **Virtual Reality/Gaming** - Natural controls
- **Sign Language** - Foundation for interpretation systems

### Learning Outcomes:
✅ Complete ML pipeline (data → train → deploy)
✅ Computer vision with OpenCV
✅ Deep learning with TensorFlow/Keras
✅ Data augmentation techniques
✅ Model evaluation and metrics
✅ Real-time inference deployment

---

## 🎓 Student-Friendly Code

This project is designed for **Intro to AI** level:

### Code Philosophy:
- ✅ **Simple, clear comments** explaining each step
- ✅ **No complex classes** - straightforward procedural code
- ✅ **Easy-to-follow logic** - linear, step-by-step approach
- ✅ **Student-level explanations** - why, not just what

### Example from `collect_data.py`:
```python
# Open webcam
camera = cv2.VideoCapture(0)

# Collect data for each gesture
for gesture in gestures:
    count = 0
    while count < samples_per_gesture:
        # Read frame from camera
        ret, frame = camera.read()

        # Convert to grayscale
        gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Resize to consistent size
        resized = cv2.resize(blurred, (200, 200))

        # Save image
        cv2.imwrite(filepath, resized)
        count += 1
```

---

## 🔧 Configuration

Edit `config.py` to customize:
```python
# Data Collection
SAMPLES_PER_GESTURE = 200
IMAGE_SIZE = (64, 64)

# Data Preprocessing
AUGMENTATION_FACTOR = 3
TEST_SIZE = 0.2

# Model Training
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
```

---

## 🚨 Troubleshooting

### Camera not opening?
```python
# Try different camera indices
camera = cv2.VideoCapture(1)  # or 2, 3, etc.
```

### Low accuracy?
- Collect more samples (300-500 per gesture)
- Ensure consistent lighting
- Vary hand positions during collection
- Check for background clutter

### Model not loading?
```bash
# Verify file exists
ls -lh models/gesture_model.h5

# Check TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 📝 Presentation Guide

### For 5-Minute Demo:

**1. Introduction (30 seconds)**
"I built a real-time hand gesture recognition system using computer vision and CNN that achieves 99.83% accuracy."

**2. Live Demo (2 minutes)**
- Run `python src/predict_live.py`
- Show all 5 gestures
- Highlight real-time performance

**3. Process Explanation (1.5 minutes)**
- Collected 2,000 images
- Applied augmentation → 6,000 images
- Trained CNN with 3 convolutional layers
- Achieved 99.83% accuracy

**4. Show Code (1 minute)**
- Open `src/train_model.ipynb`
- Highlight CNN architecture
- Explain simple, student-friendly approach

**5. Q&A (30 seconds)**

### Key Points to Mention:
- Complete ML pipeline from scratch
- Data augmentation prevents overfitting
- CNN automatically learns features
- Real-time deployment works smoothly

---

## 🔮 Future Improvements

**Short-term:**
- [ ] Add more gesture classes (6-10 gestures)
- [ ] Collect data from multiple users
- [ ] Implement background subtraction
- [ ] Fine-tune confidence thresholds

**Long-term:**
- [ ] Dynamic gesture recognition (motion-based)
- [ ] Multi-hand detection
- [ ] 3D hand pose estimation
- [ ] Integration with MediaPipe
- [ ] Mobile deployment (TensorFlow Lite)

---

## ⚠️ Limitations

**Current limitations:**
1. **Single hand** - Only recognizes one hand at a time
2. **Plain background** - Works best without clutter
3. **Consistent lighting** - Sensitive to lighting changes
4. **Static gestures** - No motion/dynamic gestures
5. **Fixed ROI** - Hand must be in designated area

**Potential improvements:**
- Use background subtraction for any background
- Implement hand tracking for ROI-free detection
- Add temporal modeling for dynamic gestures

---

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - feel free to use for educational purposes.

---

## 👤 Author

**Arham**
- Course: Introduction to AI
- Date: November 2025

---

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- TensorFlow/Keras for deep learning framework
- Scikit-learn for data processing utilities

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for learning AI/ML**

