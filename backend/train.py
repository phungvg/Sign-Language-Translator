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
LETTER_DATA_DIR = './data/asl_alphabet_train/asl_alphabet_train'
DIGIT_DATA_DIR = './data/ASL Digits/asl_dataset_digits'

LETTER_DATASET_PICKLE = './data/letter_dataset.pickle'
DIGIT_DATASET_PICKLE = './data/digit_dataset.pickle'

LETTER_MODEL_PATH = './classifier/classify_letter_model.p'
DIGIT_MODEL_PATH = './classifier/classify_number_model.p'

# Keep alphabet as default for current testing flow.
data_dir = LETTER_DATA_DIR

samples_per_class = 500
random_seed = 42

all_classes = sorted(os.listdir(data_dir))
print(f'Found {len(all_classes)} class folders: {all_classes}')

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

    # Normalize around wrist landmark (index 0) so wrist sits at (0, 0)
    # This makes the feature vector position-independent
    wrist_x = x_coords[0]
    wrist_y = y_coords[0]
    x_norm = [x - wrist_x for x in x_coords]
    y_norm = [y - wrist_y for y in y_coords]

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


#--------------------------------------------------------------------------------------------------------------------------------------------
"""Build Dataset"""
#--------------------------------------------------------------------------------------------------------------------------------------------
def build_dataset(data_dir, class_list, samples_per_class, hands_detector):
    """
    Loop over all class folders and extract landmarks from images.
    
    Args:
        data_dir: root folder containing class subdirectories
        class_list: list of class folder names (e.g., ['A', 'B', ..., 'Z'])
        samples_per_class: max images per class to sample
        hands_detector: MediaPipe hands object
    
    Returns:
        X: numpy array of shape (num_samples, 42)
        y: numpy array of shape (num_samples,) with class indices
        label_map: dict mapping class_index -> class_name for later decoding
    """
    X = []
    y = []
    label_map = {idx: classname for idx, classname in enumerate(class_list)}
    
    total_skipped = 0
    
    for class_idx, class_name in enumerate(class_list):
        class_path = os.path.join(data_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        # Get all images in this class folder
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sample randomly if more than samples_per_class
        if len(image_files) > samples_per_class:
            image_files = random.sample(image_files, samples_per_class)
        
        print(f"  Processing class '{class_name}' ({len(image_files)} images)...")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            features = extract_landmarks_from_image(img_path, hands_detector)
            
            if features is not None:
                X.append(features)
                y.append(class_idx)
            else:
                total_skipped += 1
    
    print(f"\nDataset summary: {len(X)} samples extracted, {total_skipped} images skipped (no hand detected)")
    
    return np.array(X), np.array(y), label_map


#--------------------------------------------------------------------------------------------------------------------------------------------
"""Train and Evaluate Model"""
#--------------------------------------------------------------------------------------------------------------------------------------------
def train_and_evaluate(X, y, label_map, model_output_path, report_name=""):
    """
    Train a Random Forest classifier, evaluate on train/test split.
    
    Args:
        X: feature array (N, 42)
        y: label array (N,)
        label_map: dict mapping class_idx -> class_name
        model_output_path: where to save the .p pickle file
        report_name: optional name prefix for console output (e.g., "Letter" or "Digit")
    """
    print(f"\n{'='*80}")
    print(f"Training {report_name} Model")
    print(f"{'='*80}")
    
    # Split data. Stratified split can fail when any class has <2 samples.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_seed, stratify=y
        )
    except ValueError as exc:
        print("\nWARNING: Stratified split failed. Falling back to non-stratified split.")
        print(f"Reason: {exc}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_seed, stratify=None
        )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1)
    print("\nTraining Random Forest (this may take a minute)...")
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    # Evaluate
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix (Test Set):")
    all_labels = list(label_map.keys())
    cm = confusion_matrix(y_test, y_pred_test, labels=all_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_map.values()))
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        y_test,
        y_pred_test,
        labels=all_labels,
        target_names=list(label_map.values()),
        zero_division=0,
    ))
    
    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump({'model': clf, 'label_map': label_map}, f)
    
    print(f"\nModel saved to: {model_output_path}")
    
    return clf, label_map


#--------------------------------------------------------------------------------------------------------------------------------------------
"""Main Training Pipeline"""
#--------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    # Determine which dataset(s) to train on
    mode = "both"  # Options: "letters", "digits", "both"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    print(f"\nTraining mode: {mode}")
    print(f"Random seed: {random_seed}")
    print(f"Samples per class (max): {samples_per_class}\n")
    
    # Train letters
    if mode in ["letters", "both"]:
        print("Building letter dataset...")
        letter_classes = sorted(os.listdir(LETTER_DATA_DIR))
        X_letters, y_letters, letter_map = build_dataset(
            LETTER_DATA_DIR, letter_classes, samples_per_class, hands
        )
        
        if len(X_letters) > 0:
            clf_letters, _ = train_and_evaluate(X_letters, y_letters, letter_map, LETTER_MODEL_PATH, "Letter")
        else:
            print("ERROR: No samples extracted for letters!")
    
    # Train digits
    if mode in ["digits", "both"]:
        print("\n\nBuilding digit dataset...")
        digit_classes = sorted(os.listdir(DIGIT_DATA_DIR))
        X_digits, y_digits, digit_map = build_dataset(
            DIGIT_DATA_DIR, digit_classes, samples_per_class, hands
        )
        
        if len(X_digits) > 0:
            clf_digits, _ = train_and_evaluate(X_digits, y_digits, digit_map, DIGIT_MODEL_PATH, "Digit")
        else:
            print("ERROR: No samples extracted for digits!")
    
    # Cleanup
    hands.close()
    print(f"\nTraining complete. MediaPipe closed.")