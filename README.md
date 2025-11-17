# GestureFlow - Hand Gesture Recognition

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-Educational-green)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-success)

A machine learning project that recognizes hand gestures in real-time using your webcam. Made for my Intro to AI class.

## 🎥 Demo

**[📹 Watch Demo Video](#)** *(Coming soon - record a demo video and add link here)*

**[📖 Try It Yourself](DEMO.md)** - Follow the quick start guide to run it locally

## What Does This Project Do?

This system can recognize 5 hand gestures through your webcam:
- **Fist** - make a closed fist
- **Open Palm** - open your hand completely
- **Thumbs Up** - give a thumbs up
- **Peace Sign** - make a V with two fingers
- **Pointing Finger** - point with your index finger

Just show your hand to the camera and it tells you which gesture you're making!

## Why I Built This

There are lots of situations where hands-free control is useful:
- When your hands are busy with something else
- In hospitals where touching things spreads germs
- Helping people with disabilities interact with computers
- It's also just cool to control things without touching them!

## How It Works

The project has 4 main steps:

### Step 1: Collect Data
I used my webcam to record samples of each gesture. MediaPipe detects my hand and saves the positions of 21 points on my hand (like fingertips, knuckles, wrist). I collected about 400-500 samples for each gesture under different lighting and angles.

### Step 2: Prepare the Data
The program reads all the samples I collected and splits them into:
- **Training data** (80%) - used to teach the model
- **Testing data** (20%) - used to check if the model learned correctly

### Step 3: Train the Model
I used a Random Forest classifier (a type of machine learning algorithm) to learn the patterns. It looks at the 21 hand points and learns which positions match which gestures.

### Step 4: Test in Real-Time
The trained model runs on live webcam video. It detects my hand, gets the 21 points, and predicts what gesture I'm making. It also shows how confident it is (like 95% sure it's a fist).

## My Results

- **Accuracy**: 100% (way better than the 85-90% goal!)
- **Speed**: 20+ frames per second (runs smoothly in real-time)
- **All gestures** recognized perfectly with no confusion between them

## What I Used

**Programming Language:**
- Python 3.12

**Libraries:**
- **OpenCV** - to access the webcam and process video
- **MediaPipe** - Google's tool to detect hands and get the 21 landmark points
- **scikit-learn** - for the Random Forest machine learning model
- **NumPy** and **Pandas** - for handling data
- **Matplotlib** - for creating graphs and visualizations

## Project Structure

```
GestureFlow/
├── data/
│   ├── raw/           # 2,100 gesture samples I collected (JSON files)
│   └── processed/     # Training and testing data (CSV files)
├── models/            # The trained model and performance graphs
├── src/
│   ├── data_collection/     # Script to collect gesture samples
│   ├── preprocessing/       # Script to prepare the data
│   ├── training/            # Script to train the model
│   └── inference/           # Script for real-time recognition
├── docs/              # Project proposal document
├── requirements.txt   # List of libraries needed
└── README.md          # This file
```

## How to Run This Project

### Installation

1. **Navigate to the project folder:**
```bash
cd GestureFlow
```

2. **Create a virtual environment** (keeps all libraries organized):
```bash
python3.12 -m venv venv
source venv/bin/activate
```

3. **Install required libraries:**
```bash
pip install -r requirements.txt
```

### Running the Project

#### If you want to train your own model:

**1. Collect your gesture data:**
```bash
python src/data_collection/collect_gestures.py
```
- Press SPACE to start collecting samples
- The program will tell you which gesture to make
- Collect 100+ samples for each gesture
- Press Q to quit

**2. Prepare the data:**
```bash
python src/preprocessing/prepare_dataset.py
```

**3. Train the model:**
```bash
python src/training/train_model.py
```
This creates the trained model and shows you how accurate it is.

**4. Test it live:**
```bash
python src/inference/real_time_inference.py
```
Make gestures and watch it recognize them in real-time!

#### If you just want to see it work:

Run step 4 above with my pre-trained model (if you have my data and model files).

## What I Learned

- How to collect and prepare data for machine learning
- Using MediaPipe for computer vision tasks
- Training and evaluating machine learning models
- The difference between training accuracy and testing accuracy
- How to make predictions in real-time
- Reading confusion matrices to see which gestures get mixed up

## Challenges I Faced

1. **Camera quality** - My phone's rear camera was broken so I had to use Camo to stream my phone's front camera
2. **Python version** - MediaPipe doesn't work with Python 3.13 yet, had to use Python 3.12
3. **Collecting good data** - Needed to vary lighting and angles to make the model work in different conditions

## Possible Improvements

- Add more gestures (thumbs down, OK sign, etc.)
- Recognize gestures with both hands
- Use the gestures to control something (like volume or a game)
- Try deep learning models like CNNs
- Make it work on a phone

## Technical Notes

### How the Features Work
- MediaPipe detects 21 points on your hand
- Each point has x, y, and z coordinates (3D position)
- Total: 21 points × 3 coordinates = 63 features
- All positions are normalized relative to the wrist so hand size doesn't matter

### Why Random Forest?
- Works well with small datasets
- Fast enough for real-time predictions
- Easy to understand and visualize
- Good accuracy without needing lots of computing power

### Performance Metrics
- **Accuracy**: How many predictions were correct
- **Precision**: Out of all "fist" predictions, how many were actually fists
- **Recall**: Out of all actual fists, how many did we find
- **F1-Score**: Balance between precision and recall
- **Confusion Matrix**: Shows which gestures get confused with each other

## Files Generated

After running the project, you'll have:
- **gesture_classifier.joblib** - the trained model
- **confusion_matrix.png** - visualization showing classification results
- **feature_importance.png** - which hand points are most important
- **train.csv** and **test.csv** - the prepared datasets

## Troubleshooting

**"No module named cv2"**
- Make sure you activated the virtual environment: `source venv/bin/activate`

**Camera not working**
- Check if another app is using the camera
- Make sure you gave permission to access the camera
- Try unplugging and replugging webcam

**Low FPS or slow performance**
- Close other programs
- Try reducing the number of trees in Random Forest (change 100 to 50)

**Gestures not recognized well**
- Make sure there's good lighting
- Keep your hand in the camera frame
- Try recollecting data with more variety

## Credits

- **MediaPipe** by Google - for hand detection
- **scikit-learn** - for machine learning tools
- **OpenCV** - for camera and video processing

---

**Note:** This project was made for educational purposes as part of my Introduction to AI course.