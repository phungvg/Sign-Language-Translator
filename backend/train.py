import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Data path"""
#--------------------------------------------------------------------------------------------------------------------------------------------
letter_data_dir    = './dataset/archive/asl_alphabet_train/'
digit_data_dir    = './dataset/archive/asl_number_train/'

#Folder for extracted landmarks, only saves the 42 numbers, not the img
letter_dataset_pickle = "./dataset/archive/letter_dataset_pickle" 
digit_dataset_pickle = "./dataset/archive/digit_dataset_pickle" 

#Save the model
classifier_dir = './classifier'

#Letter and number model
letter_model   = './classifier/classify_letter_model.p'
digit_model   = './classifier/classify_digit_model.p'

samples_per_class = 500

DATA_CONFIGS = [
    {
        'name': 'letter',
        'data_dir': letter_data_dir,
        'dataset_pickle': letter_dataset_pickle,
        'model_path': letter_model,
        'preview_class': 'A',
    },
    {
        'name': 'digit',
        'data_dir': digit_data_dir,
        'dataset_pickle': digit_dataset_pickle,
        'model_path': digit_model,
        'preview_class': '0',
    },
]

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
"""Extract landmarks from images
P0: Wrist
P1: Thumb CMC (base of thump)
P2: Thump MCP (mid thump)
P3: Thump IP (Sit partially behind P4/P5)
P4: Thumb Tip
P5: Index Finger MCP (Base of the index finger)
Normalized (x,y) coordinates by MediaPipe (0-1) in the frame
"""
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
"""Visualize landmark"""
#--------------------------------------------------------------------------------------------------------------------------------------------
def visualize_landmarks(img_path, hands, num_points=6):
    """
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


def resolve_dataset_pickle_path(path):
    # If no filename is provided, use dataset.pickle inside the directory.
    if path.endswith('.p') or path.endswith('.pickle'):
        return path
    return os.path.join(path, 'dataset.pickle')


def find_preview_image(data_dir, preferred_class):
    class_order = [preferred_class]
    class_order.extend(
        [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d)) and d != preferred_class]
    )

    for class_name in class_order:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        image_files = sorted(
            [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        )
        if image_files:
            return os.path.join(class_dir, image_files[0])

    return None

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Process the whole data for extract landmarks"""
#--------------------------------------------------------------------------------------------------------------------------------------------
def process_and_save_dataset(data_dir, dataset_pickle, dataset_name):
    data = []
    labels = []

    all_classes = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )
    print(f"[{dataset_name}] Found {len(all_classes)} class folders: {all_classes}")
    print(f"[{dataset_name}] Starting feature extraction for {len(all_classes)} classes...")
    
    for class_name in all_classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Get all images in this class folder
        image_files = sorted(os.listdir(class_dir))
        
        # Limit to samples_per_class to balance dataset and speed up extraction
        if len(image_files) > samples_per_class:
            image_files = image_files[:samples_per_class]
            
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
    load_path = resolve_dataset_pickle_path(dataset_pickle)
        
    print(f"\n[{dataset_name}] Loading dataset from {load_path}...")
    try:
        data_dict = pickle.load(open(load_path, 'rb'))
    except FileNotFoundError:
        print(f"[{dataset_name}] Error: {load_path} not found. Please run feature extraction first.")
        return
        
    data = data_dict['data']
    labels = data_dict['labels']

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
     
    print(f"[{dataset_name}] Clean samples  : {len(clean_data)}")
    print(f"[{dataset_name}] Bad samples    : {bad_count} (removed)")
     
    X = np.array(clean_data)
    y = np.array(clean_labels)
     
    # Show class distribution
    print(f"\n[{dataset_name}] Samples per class:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls:20s}: {cnt}")
     
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=True,
        stratify=y,
        random_state=42
    )
     
    print(f"\n[{dataset_name}] Training samples : {len(X_train)}")
    print(f"[{dataset_name}] Testing samples  : {len(X_test)}")
     
    # Train
    print(f"\n[{dataset_name}] Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
     
    # Evaluate
    y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[{dataset_name}] Accuracy: {accuracy:.2%}")
     
    print(f"\n[{dataset_name}] Classification Report:")
    print(classification_report(y_test, y_pred))
     
    # Ensure classifier directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump({'model': clf}, f)
     
    print(f"[{dataset_name}] Model saved to {model_path}")
    """Confusion matrix """
    # Generate and display confusion matrix
    label_classes = sorted(list(set(y_test)))
     
    cm   = confusion_matrix(y_test, y_pred, labels=label_classes)
    fig, ax = plt.subplots(figsize=(10, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_classes)
    disp.plot(cmap=plt.cm.Blues, values_format='g', ax=ax)
    plt.title(f'{dataset_name.title()} Model - Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    cm_path = f'./classifier/confusion_matrix_{dataset_name}.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.show()
     
    print(f"\n[{dataset_name}] Confusion matrix saved to {cm_path}")
    print(f"\n[{dataset_name}] All done! Model ready at: {model_path}")
    print("Next step: run main.py")

# --------------------------------------------------------------------------------------------------------------------------------------------
"""Main"""
#--------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    for cfg in DATA_CONFIGS:
        print(f"\n{'=' * 72}")
        print(f"Running pipeline for: {cfg['name']}")
        print(f"{'=' * 72}")

        # Test on 1 image before full extraction for quick visual sanity check.
        preview_img = find_preview_image(cfg['data_dir'], cfg['preview_class'])
        if preview_img:
            print(f"[{cfg['name']}] Previewing landmarks on: {preview_img}")
            visualize_landmarks(preview_img, hands, num_points=6)
        else:
            print(f"[{cfg['name']}] No preview image found, skipping landmark visualization.")

        process_and_save_dataset(
            data_dir=cfg['data_dir'],
            dataset_pickle=cfg['dataset_pickle'],
            dataset_name=cfg['name'],
        )

        train_classifier(
            dataset_pickle=cfg['dataset_pickle'],
            model_path=cfg['model_path'],
            dataset_name=cfg['name'],
        )