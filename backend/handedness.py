"""
Use 2 models from train: 
Digit : classify_digit_model.p  -> Right hand
Letter: classify_letter_model.p -> Left hand

For current testing: use dummy result, no webcam neededm simulate a right hand detection
"""
import os
import mediapipe as mp
import cv2
import threading
import numpy
from postprocess import DebounceBuffer, suggest_completions 

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Load from utils.py"""
#--------------------------------------------------------------------------------------------------------------------------------------------
from utils import (
    load_model,                  # load .p files
    get_handedness,              # left or right?
    extract_landmarks_by_index,  # get 42 numbers for specific hand
    normalize_landmarks,         # make position-independent
    get_wrist_x                  # sort two hands left-to-right
)

#--------------------------------------------------------------------------------------------------------------------------------------------
"""Load 2 models"""
#--------------------------------------------------------------------------------------------------------------------------------------------
def load_models():
    letter_model = load_model('classifier/classify_letter_model.p')
    digit_model  =load_model('classifier/classify_digit_model.p')
    return letter_model, digit_model


#--------------------------------------------------------------------------------------------------------------------------------------------
"""HandRouter class
-Frames comes in
-Check #hands (0,1,2)
    + Each hand: L or R
    + Extract 42 landmarks
    + Normalize
    + Route to correct model
    + Get prediction + confidene
- Return list of predictions

"""
#--------------------------------------------------------------------------------------------------------------------------------------------
class HandRouter:
    """
    Routes each detected hand to the correct model
    R: letter (A-Z), del,space
    L: digit  (0-9)
    """

    def __init__(self,letter_model, digit_model):
        self.letter_model = letter_model
        self.digit_model  = digit_model

    
    def predict(self,results):
        """Call in every frame
        Process all detected hands and returns predictions

        Args:
            results: MediaPipe Hands process() output from current frame
            
        Returns:
            list of dicts, one per hand detected:
            [
                {
                    "label"     : "A",        ← predicted class
                    "type"      : "letter",   ← "letter" or "digit"
                    "hand"      : "Right",    ← which hand
                    "confidence": 0.91        ← how confident (0.0-1.0)
                },
                ...
            ]
            Empty list [] if no hands detected.
        """
        predictions = []

        # Get handedness dict {0: "Right"} or {0: "Left", 1: "Right"}
        handedness = get_handedness(results)

        if not handedness:
            return []  # no hands detected

        # Sort hands left-to-right by wrist x position
        # So output order is always consistent
        sorted_hands = sorted(
            handedness.items(),
            key=lambda x: get_wrist_x(results, x[0])
        )

        for hand_index, hand_label in sorted_hands:
            prediction = self._predict_single_hand(
                results,
                hand_index,
                hand_label
            )
            if prediction is not None:
                predictions.append(prediction)

        return predictions
    

    def _predict_single_hand(self, results, hand_index, hand_label):
        """
        Helper — process one hand and return its prediction.
        
        Steps:
            1. Extract 42 landmark values for this hand
            2. Normalize (remove position/size dependency)
            3. Route to correct model (Left=digit, Right=letter)
            4. Return prediction + confidence score

        Args:
            results    : MediaPipe output
            hand_index : 0 or 1 (which hand in results)
            hand_label : "Left" or "Right"

        Returns:
            dict with label, type, hand, confidence
            or None if landmark extraction fails
        """
    
        # Extract 42 landmark values for this specific hand
        landmarks = extract_landmarks_by_index(results, hand_index)
        if landmarks is None:
            return None

        # Normalize — remove position and size dependency
        normalized = normalize_landmarks(landmarks)

        # Route to correct model based on hand label
        if hand_label == "Right":
            model      = self.letter_model
            hand_type  = "letter"
        else:
            model      = self.digit_model
            hand_type  = "digit"

        # Get prediction
        # predict_proba gives confidence per class
        # predict gives the winning class label
        proba      = model.predict_proba([normalized])[0]
        label      = model.predict([normalized])[0]
        confidence = float(proba.max())

        return {
            "landmarks" : landmarks,
            "label"     : label,
            "type"      : hand_type,
            "hand"      : hand_label,
            "confidence": confidence
        }
    


lock = threading.Lock()

# Create global instances of models, router, and MediaPipe Hands
# Using lock to ensure thread safety while processing several frames at once
letter_model, digit_model = load_models()
router = HandRouter(letter_model, digit_model)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,      # Live video, not static images
    max_num_hands=1,              # one hand per image
    min_detection_confidence=0.5
)

debounce=DebounceBuffer() #From postprocess

def handle_frame(frame):
    with lock:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results is None:
            return {}

        predictions = router.predict(results)

        if not predictions:
            #Test proprocess
            # No hand detected → reset debounce (brief rest)
            debounce.reset()
            return {}
        
        pred = predictions[0]
        label = pred["label"]

        # Run through debounce
        clean_label = debounce.push(label)

        if clean_label is None:
            return {}  # suppressed duplicate

        pred["label"] = clean_label
        
        # return predictions[0]
        return pred

#--------------------------------------------------------------------------------------------------------------------------------------------
"""MAIN"""
#--------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Run this to verify handedness.py works before connecting to main.py.
    No webcam needed — tests model loading and router setup only.
    
    Usage:
        python handedness.py
    """
    print("=" * 60)
    print("handedness.py self-test")
    print("=" * 60)

    #  Load models 
    print("\n Loading models...")
    try:
        letter_model, digit_model = load_models()
        print("  letter_model : OK")
        print("  digit_model  : OK")
    except FileNotFoundError as e:
        print(f"  ERROR: Model file not found — {e}")
        print("  Run train.py first to generate model files.")
        exit(1)

    # Create HandRouter 
    try:
        router = HandRouter(letter_model, digit_model)
        print("  HandRouter   : OK")
    except Exception as e:
        print(f"  ERROR: {e}")
        exit(1)

    #  Predict with no hand 
    print("\n Predict with no hand detected...")

    class FakeResults:
        """Simulates MediaPipe results with no hand detected."""
        multi_hand_landmarks = None
        multi_handedness     = None

    result = router.predict(FakeResults())
    assert result == [], f"Expected [] but got {result}"
    print("  No hand [] : OK")

    # Check model classes 
    print("\nChecking model classes...")
    letter_classes = letter_model.classes_
    digit_classes  = digit_model.classes_

    print(f"  Letter classes ({len(letter_classes)}): {sorted(letter_classes)}")
    print(f"  Digit classes  ({len(digit_classes)}): {sorted(digit_classes)}")

    # Make sure expected classes are present
    expected_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]
    expected_digits  = [str(i) for i in range(10)]

    for cls in expected_letters:
        assert cls in letter_classes, f"Missing letter class: {cls}"
    for cls in expected_digits:
        assert cls in digit_classes, f"Missing digit class: {cls}"

    print("  All expected classes present : OK")

    # 
    print("\n" + "=" * 60)
    print("All tests passed. handedness.py is ready.")
    print("=" * 60)