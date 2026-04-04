import os
import cv2
import pickle
import random
import sys
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Data path"""
#--------------------------------------------------------------------------------------------------------------------------------------------
LETTER_DATA_DIR   = './dataset/archive/asl_alphabet_train/'
DIGIT_DATA_DIR    = './dataset/archive/asl_digit_train/'

#Folder for extracted landmarks, only saves the 42 numbers, not the img
LETTER_DATASET_PICKLE = "./dataset/archive/letter_dataset.pickle" 
DIGIT_DATASET_PICKLE  = "./dataset/archive/digit_dataset.pickle" 

#Save the model output
CLASSIFIER_DIR = './classifier'

#Letter and number model
LETTER_MODEL   = './classifier/classify_letter_model.p'
DIGIT_MODEL    = './classifier/classify_digit_model.p'

#Training config
random_seed = 42

# Pipeline config — controls which datasets get processed

DATA_CONFIGS = [
    {
        'name': 'letter',
        'data_dir': LETTER_DATA_DIR,
        'dataset_pickle': LETTER_DATASET_PICKLE,
        'model_path': LETTER_MODEL,
        'preview_class': 'A', #class used for quick check
        'samples_per_class' : 3000
    },
    {
        'name': 'digit',
        'data_dir': DIGIT_DATA_DIR,
        'dataset_pickle': DIGIT_DATASET_PICKLE,
        'model_path': DIGIT_MODEL,
        'preview_class': '0',  #class used for quick check
        'samples_per_class' : 500 
    },
]

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Setup MediaPipe"""
#--------------------------------------------------------------------------------------------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # True = images, not video
    max_num_hands=1,              # one hand per image
    min_detection_confidence=0.3  # lower = detects more hands but less confident
)

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Extract landmarks from images
P0: Wrist
P1: Thumb CMC (base of thump)
P2: Thump MCP (mid thump)
P3: Thump IP (Sit partially behind P4/P5)
P4: Thumb Tip
P5: Index Finger MCP (Base of the index finger)
P6: Index PIP
P7: Index DIP
P8: Index Tip
P9: Middle MCP
   ...
P20: Pinky Tip

Each landmark gives (x, y) normalized 0.0-1.0 in the frame.
We save x and y only (drop z) -> 21 x 2 = 42 values per image.
"""
#--------------------------------------------------------------------------------------------------------------------------------------------
"""
Go thr every img, run MediaPipe and saves 42 (2*x,y =42) landmark values (x,y). No z coordinate 

Normalization:
        Subtract the minimum x and y so the hand position in the frame
        does not affect the feature vector. Same sign anywhere = same numbers.
 
    Args:
        img_path      : path to image file
        hands_detector: initialized MediaPipe Hands object
"""

def extract_landmarks_from_image(img_path, hands):
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
"""Visualize landmark"""
#--------------------------------------------------------------------------------------------------------------------------------------------
def visualize_landmarks(img_path, hands, num_points=6):
    """ 
    Quick check
    Visualizes the first N hand landmarks on the original image.
    Each landmark corresponds to an (x, y) coordinate pair in the feature vector.
    For example, showing 6 points corresponds to the first 12 values in the feature vector.
    
    Args:
        img_path (str): Path to the image file.
        hands: Initialized MediaPipe hands object.
        num_points (int): The number of landmarks to display (default is 6).
    """
    img = cv2.imread(img_path)
    if img is None:
        print("Could not read image for visualization.")
        return

    # Convert to RGB for MediaPipe and Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        h, w, _ = img.shape
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_rgb)
        
        # Display the specified number of landmarks
        for i in range(min(num_points, len(hand.landmark))):
            lm = hand.landmark[i]
            cx, cy = int(lm.x * w), int(lm.y * h)
            
            # Plot the point using matplotlib
            plt.scatter(cx, cy, c='blue', s=40, zorder=5)
            
            # Add text annotation
            label_text = f"P{i}: ({round(lm.x, 3)}, {round(lm.y, 3)})"
            # bbox makes text readable over any background color
            plt.annotate(
                label_text, 
                (cx, cy),
                xytext=(cx + 10, cy),
                fontsize=8,
                color='red',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                zorder=10
            )
            
        plt.title(f"First {num_points} Landmarks")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("No hand detected during visualization.")

    # Also print the feature vector for inspection
    features = extract_landmarks_from_image(img_path, hands)
    if features:
        print(f"Feature vector length : {len(features)} (should be 42)")
        print(f"First 6 values        : {[round(v, 4) for v in features[:6]]}")
        print(f"  P0 wrist      x={round(features[0], 4)}, y={round(features[1], 4)}")
        print(f"  P1 thumb CMC  x={round(features[2], 4)}, y={round(features[3], 4)}")
        print(f"  P2 thumb MCP  x={round(features[4], 4)}, y={round(features[5], 4)}")


def resolve_dataset_pickle_path(path):
    # If no filename is provided, use dataset.pickle inside the directory.
    if path.endswith('.p') or path.endswith('.pickle'):
        return path
    return os.path.join(path, 'dataset.pickle')


def find_preview_image(data_dir, preferred_class):
    """
    Find one image to use for the quick visual sanity test.
    Tries the preferred class first, then falls back to any other class.
    """
    class_order = [preferred_class]
    class_order.extend(
        [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d)) and d != preferred_class]
    )

    for class_name in class_order:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        image_files = sorted([
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
            and not f.startswith('.')  
        ])

        if image_files:
            return os.path.join(class_dir, image_files[0])

    return None

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Process the whole data for extract landmarks"""
#--------------------------------------------------------------------------------------------------------------------------------------------
def process_and_save_dataset(data_dir, dataset_pickle, dataset_name, samples_per_class):
    """
    Loop over all class folders, extract landmarks from images, save results to a pickle file.

    Pickle:
        Extracting landmarks is slow (5-10 mins for thousands of images).
        Saving to pickle means you only do it once. Retraining from saved
        pickle takes seconds instead of minutes.

    Args:
        data_dir       : root folder containing class subfolders
        dataset_pickle : output path for the pickle file
        dataset_name   : label for console output ("letter" or "digit")
    """
    data = []
    labels = []

    all_classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and not d.startswith('.')
    ])

    print(f"[{dataset_name}] Found {len(all_classes)} class folders: {all_classes}")
    print(f"[{dataset_name}] Max samples per class: {samples_per_class}")
    print(f"[{dataset_name}] Starting feature extraction for {len(all_classes)} classes...")
    
    total_skipped = 0

    for class_name in all_classes:
        class_dir = os.path.join(data_dir, class_name)
        image_files = [
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
            and not f.startswith('.')
        ]
            
        
        # Limit to samples_per_class to balance dataset and speed up extraction
        if len(image_files) > samples_per_class:
            image_files = random.sample(image_files, samples_per_class)
            
        print(f"[{dataset_name}] Processing class '{class_name}': {len(image_files)} images...")
        
        success_count = 0
        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            features = extract_landmarks_from_image(img_path, hands)
            
            # If a hand was successfully detected and features extracted
            if features is not None:
                data.append(features)
                labels.append(class_name)
                success_count += 1
                
        print(f"[{dataset_name}]   -> Extracted features from {success_count}/{len(image_files)} images.")

    # Save to pickle so we can use it to train the model later
    print(f"\n[{dataset_name}] Total extracted samples: {len(data)}")

    save_path = resolve_dataset_pickle_path(dataset_pickle)
        
    print(f"[{dataset_name}] Saving dataset to {save_path}...")
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        with open(save_path, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        print(f"[{dataset_name}] Dataset saved successfully!")
    except Exception as e:
        print(f"[{dataset_name}] Failed to save dataset: {e}")

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Train the classifier"""
#--------------------------------------------------------------------------------------------------------------------------------------------
def train_classifier(dataset_pickle, model_path, dataset_name):
    """
    Load extracted landmarks from pickle, train Random Forest,
    evaluate, save model and confusion matrix.

    Args:
        dataset_pickle : path to the pickle file from build_dataset()
        model_path     : where to save the trained model .p file
        dataset_name   : label for console output ("letter" or "digit")
    """
    print(f"\n[{dataset_name}] Loading dataset from {dataset_pickle}...")

    load_path = resolve_dataset_pickle_path(dataset_pickle)

    if not os.path.exists(load_path):
        print(f"[{dataset_name}] ERROR: {dataset_pickle} not found.")
        print(f"[{dataset_name}] Run build_dataset() first.")
        return

    with open(load_path, 'rb') as f:
        dataset = pickle.load(f)

    data   = dataset['data']
    labels = dataset['labels']

    # Sanity check — every vector must be exactly 42
    clean_data   = []
    clean_labels = []
    bad_count    = 0

    for vec, lbl in zip(data, labels):
        if len(vec) == 42:
            clean_data.append(vec)
            clean_labels.append(lbl)
        else:
            bad_count += 1

    print(f"[{dataset_name}] Clean samples : {len(clean_data)}")
    print(f"[{dataset_name}] Bad samples   : {bad_count} removed")

    X = np.array(clean_data)
    y = np.array(clean_labels)

    # Show class distribution
    print(f"\n[{dataset_name}] Samples per class:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls:20s}: {cnt}")

    # Train/test split — stratified to ensure all classes in both sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            shuffle=True,
            stratify=y,
            random_state=random_seed
        )
    except ValueError as e:
        print(f"\n[{dataset_name}] WARNING: Stratified split failed — {e}")
        print(f"[{dataset_name}] Falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            shuffle=True,
            random_state=random_seed
        )

    print(f"\n[{dataset_name}] Training samples : {len(X_train)}")
    print(f"[{dataset_name}] Testing samples  : {len(X_test)}")

    # Train Random Forest
    # n_jobs=-1 uses all CPU cores — faster training
    print(f"\n[{dataset_name}] Training Random Forest (n_estimators=100)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=random_seed,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred     = clf.predict(X_test)
    train_acc  = accuracy_score(y_train, clf.predict(X_train))
    test_acc   = accuracy_score(y_test, y_pred)

    print(f"\n[{dataset_name}] Train accuracy : {train_acc:.2%}")
    print(f"[{dataset_name}] Test accuracy  : {test_acc:.2%}")

    print(f"\n[{dataset_name}] Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model — also save label map so main.py knows class names
    os.makedirs(CLASSIFIER_DIR, exist_ok=True)
    label_map = {i: cls for i, cls in enumerate(sorted(set(y)))}

    with open(model_path, 'wb') as f:
        pickle.dump({'model': clf, 'label_map': label_map}, f)

    print(f"[{dataset_name}] Model saved to {model_path}")

    # Confusion matrix
    label_classes = sorted(list(set(y_test)))
    cm   = confusion_matrix(y_test, y_pred, labels=label_classes)
    fig, ax = plt.subplots(figsize=(14, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_classes)
    disp.plot(cmap=plt.cm.Blues, values_format='g', ax=ax)
    plt.title(f'{dataset_name.title()} Model — Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()

    cm_path = f'./classifier/confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"[{dataset_name}] Confusion matrix saved to {cm_path}")
    print(f"[{dataset_name}] Done! Model ready at: {model_path}\n")


#--------------------------------------------------------------------------------------------------------------------------------------------
"""MAIN"""
#--------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":    
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    print(f"\n{'=' * 72}")
    print(f"ASL Fingerspelling Training Pipeline")
    print(f"Mode            : {mode}")
    print(f"Random seed     : {random_seed}")
    print(f"{'=' * 72}")

    for cfg in DATA_CONFIGS:

        # Skip if mode doesn't match
        if mode != "both" and mode != cfg['name']:
            continue

        # Check data folder exists
        if not os.path.exists(cfg['data_dir']):
            print(f"\n[{cfg['name']}] WARNING: Data folder not found: {cfg['data_dir']}")
            print(f"[{cfg['name']}] Skipping {cfg['name']} pipeline.")
            continue

        print(f"Pipeline: {cfg['name'].upper()}")
    

        # Quick visual sanity test 

        # print(f"\n[{cfg['name']}] Quick sanity test on one image")
        # preview_img = find_preview_image(cfg['data_dir'], cfg['preview_class'])

        # if preview_img:
        #     print(f"[{cfg['name']}] Testing on: {preview_img}")
        #     visualize_landmarks(preview_img, hands, num_points=6)
        # else:
        #     print(f"[{cfg['name']}] No preview image found, skipping visualization.")
        
        #-----------------------------------------------------------------------------------
        #  Extract full landmarks from all images 

        print(f"\n[{cfg['name']}] Extracting landmarks from all images")
        print(f"[{cfg['name']}] Extracting...")

        process_and_save_dataset(
            data_dir         = cfg['data_dir'],
            dataset_pickle   = cfg['dataset_pickle'],
            dataset_name     = cfg['name'],
            samples_per_class= cfg['samples_per_class']
        )

        #  Train + evaluate + save model 
        print(f"\n[{cfg['name']}] Training classifier")

        train_classifier(
            dataset_pickle= cfg['dataset_pickle'],
            model_path    = cfg['model_path'],
            dataset_name  = cfg['name'],
        )

    # Clean up MediaPipe
    hands.close()
    print(f"All pipelines complete. MediaPipe closed.")
