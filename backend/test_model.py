"""
Test on test files
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from handedness import load_models
from utils import extract_landmarks_from_image, normalize_landmarks
#--------------------------------------------------------------------------------------------------------------------------------------------
"""Test folders"""
#--------------------------------------------------------------------------------------------------------------------------------------------
LETTER_TEST_DIR = './dataset/archive/asl_alphabet_test/'
DIGIT_TEST_DIR  = './dataset/archive/asl_digit_test/'

confidence_threshold = 0.0  # show all predictions regardless of confidence

#--------------------------------------------------------------------------------------------------------------------------------------------
"""MediaPipe"""
#--------------------------------------------------------------------------------------------------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, #Img= True, Video=False
    max_num_hands=1,
    min_detection_confidence=0.3
)

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Test function"""
#--------------------------------------------------------------------------------------------------------------------------------------------
def test_model(model, test_dir, model_name):
    """
    Test a trained model against a folder of unseen images.

    Args:
        model      : loaded Random Forest model
        test_dir   : path to test image folder
        model_name : label for console output ("letter" or "digit")
    """
    print(f"Testing {model_name.upper()} model on: {test_dir}")
    print(f"{'=' * 60}")

    if not os.path.exists(test_dir):
        print(f"ERROR: Test folder not found: {test_dir}")
        return

    # Get all class folders (A,B,C... or 0,1,2...)
    classes = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
        and not d.startswith('.')
    ])

    print(f"Found {len(classes)} classes: {classes}\n")

    correct   = 0
    total     = 0
    failed    = []   # wrong predictions
    no_hand   = []   # images where MediaPipe found no hand

    for true_label in classes:
        class_dir  = os.path.join(test_dir, true_label)
        img_files  = [
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
            and not f.startswith('.')
        ]

        #Loop thr each img in that folder
        for img_file in img_files:
            img_path = os.path.join(class_dir, img_file)
            total   += 1

            # Extract landmarks
            landmarks = extract_landmarks_from_image(img_path, hands)

            if landmarks is None:
                # MediaPipe couldn't detect a hand
                no_hand.append({
                    "true" : true_label,
                    "file" : img_file
                })
                print(f"  [{true_label}] {img_file} -> NO HAND DETECTED")
                continue

            # Normalize
            normalized = normalize_landmarks(landmarks)

            # Predict
            proba      = model.predict_proba([normalized])[0]
            predicted  = model.predict([normalized])[0]
            confidence = float(proba.max())

            if predicted == true_label:
                correct += 1
                status   = "High"
            else:
                failed.append({
                    "true"      : true_label,
                    "predicted" : predicted,
                    "confidence": confidence,
                    "file"      : img_file
                })
                status = "Low"

            print(f"  [{true_label}] {img_file:<30} -> {predicted:>6}  ({confidence:.0%})  {status}")

    detected = total - len(no_hand)
    accuracy = correct / detected if detected > 0 else 0

    print(f"\n{'─' * 60}")
    print(f"[{model_name}] Results:")
    print(f"  Total images    : {total}")
    print(f"  No hand found   : {len(no_hand)}")
    print(f"  Tested          : {detected}")
    print(f"  Correct         : {correct}")
    print(f"  Wrong           : {len(failed)}")
    print(f"  Accuracy        : {accuracy:.2%}")

    if failed:
        print(f"\n  Wrong predictions:")
        for f in failed:
            print(f"    True={f['true']:>6} -> Predicted={f['predicted']:>6}  ({f['confidence']:.0%})")

    if no_hand:
        print(f"\n  No hand detected in:")
        for f in no_hand:
            print(f"    [{f['true']}] {f['file']}")

    print(f"{'─' * 60}")
    return accuracy


#--------------------------------------------------------------------------------------------------------------------------------------------
""" MAIN"""
# #--------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    print(f"\n{'=' * 60}")
    print(f"ASL Model Test")
    print(f"Mode: {mode}")
    print(f"{'=' * 60}")

    # Load models
    print("\nLoading models...")
    try:
        letter_model, digit_model = load_models()
        print("  Models loaded OK")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Run train.py first.")
        exit(1)

    results_summary = {}

    # Test letter model
    if mode in ["both", "letters"]:
        acc = test_model(letter_model, LETTER_TEST_DIR, "letter")
        if acc is not None:
            results_summary["letter"] = acc

    # Test digit model
    if mode in ["both", "digits"]:
        acc = test_model(digit_model, DIGIT_TEST_DIR, "digit")
        if acc is not None:
            results_summary["digit"] = acc

    # Final summary
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    for name, acc in results_summary.items():
        print(f"  {name:10s} model accuracy: {acc:.2%}")

    # Cleanup
    hands.close()
    print(f"\nDone. MediaPipe closed.")