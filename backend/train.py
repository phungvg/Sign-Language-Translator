import os
import cv2
import pickle
import random
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Data path"""
#--------------------------------------------------------------------------------------------------------------------------------------------
data_dir    = './data/asl_alphabet_train/asl_alphabet_train'
data_set_path = './data/letter_dataset.pickle'
letter_moedl   = './classifier/classify_letter_model.p'
data_pickle = "./data/data.pickle"

samples_per_class = 500
random_seed = 42

all_classess = sorted(os.listdir(data_dir))
print(f'Found {len(all_classess)} class folders: {all_classess}')

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Setup MediaPipe"""
#--------------------------------------------------------------------------------------------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # True = processing images, not video
    max_num_hands=1,              # one hand per image
    min_detection_confidence=0.3  # lower = detects more hands but less confident
)

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Extract landmarks from images"""
#--------------------------------------------------------------------------------------------------------------------------------------------
"""
Go thr every img, run MediaPipe and saves 42 (2*x,y =42) landmark values (x,y). No z coordinate 
"""
def extract_landmarks_from_image(img_path, hands):
    """
    Takes an image path, runs MediaPipe, returns 42-float feature vector.

    What the 42 numbers represent:
      - 21 hand landmarks (wrist + 4 joints per finger)
      - Each landmark has x and y coordinate = 21 × 2 = 42
      - Normalized so wrist is at (0,0) — position in frame doesn't matter

    Returns:
        list of 42 floats, or None if no hand detected
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        return None

    # MediaPipe needs RGB, OpenCV loads BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run hand detection
    results = hands.process(img_rgb)

    # No hand found → skip this image
    if not results.multi_hand_landmarks:
        return None

    # Get the first (only) hand
    hand = results.multi_hand_landmarks[0]

    # Extract raw x and y coordinates (values between 0.0 and 1.0)
    x_coords = [lm.x for lm in hand.landmark]  # 21 values
    y_coords = [lm.y for lm in hand.landmark]  # 21 values

    # Normalize: subtract the minimum so the wrist sits at (0, 0)
    # This makes the feature vector position-independent
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_norm = [x - x_min for x in x_coords]
    y_norm = [y - y_min for y in y_coords]

    # Interleave x and y into one flat list:
    # [x0, y0, x1, y1, x2, y2, ... x20, y20] = 42 values
    feature_vector = []
    for x, y in zip(x_norm, y_norm):
        feature_vector.append(x)
        feature_vector.append(y)

    return feature_vector  # 42 floats



#--------------------------------------------------------------------------------------------------------------------------------------------
"""Test"""
#--------------------------------------------------------------------------------------------------------------------------------------------
test_img="./data/asl_alphabet_train/asl_alphabet_train/A/A1.jpg"
features = extract_landmarks_from_image(test_img, hands)

if features:
    print("Hand detected: True")
    print("Feature vector length:", len(features))  # should be 42
    print("First 6 values:", [round(v, 4) for v in features[:6]])
else:
    print("No hand detected")