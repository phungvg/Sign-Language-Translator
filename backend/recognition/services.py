from .utils import SignProcessor

processor = SignProcessor()

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ml.utils import build_feature_vector, load_model, get_handedness

# Create an HandLandmarker object
# NEW MediaPipe Task API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Load ML model
# letter_model = load_model('./classifier/classify_letter_model.p')
# number_model = load_model('./classifier/classify_number_model.p')

def detect_landmarks(frame):
    try: 
        # convert OpenCV BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # convert numpy to mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hand landmarks from the input image
        detection_result = detector.detect(mp_image)

        return detection_result
    except Exception as e:
        print("MediaPipe error:", e)
        return None

def predict_letter(frame):
    # detect with MediaPipe
    result = detect_landmarks(frame)

    if result is None:
        return None

    if not result.hand_landmarks:
        return None

    coords = []
    for lm in result.hand_landmarks[0]:
        coords.extend([lm.x, lm.y, lm.z])

    features = build_feature_vector(coords)

    handedness = get_handedness(result)

    # if handedness.get(0) == "Right":
    #     prediction = letter_model.predict([features])[0]
    # else:
    #     prediction = number_model.predict([features])[0]

    # return prediction
    return 'A' # remove this 

# Handle gesture prediction
def handle_frame(image):
    letter = predict_letter(image)

    if letter is None:
        return ""
    text = processor.update(letter)

    return text