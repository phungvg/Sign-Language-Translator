"""
utils.py — Utility functions for ASL Fingerspelling
----------------------------------------------------
Combines drawing/helper functions
with new feature engineering needed for improved accuracy.

Sections:
  1. Constants & Colors
  2. Model helpers  (load_model)
  3. Drawing helpers (draw_landmarks, draw_info_text, calc_landmark_list)
  4. Save helpers   (save_gif, save_video)
  5. Feature engineering —  (extract, normalize, angles, build_feature_vector)
  6. Handedness     —  (get_handedness, count_hands, get_wrist_x)
"""

import cv2
import pickle
import string
import numpy as np
import imageio

# ─────────────────────────────────────────────
# 1. CONSTANTS & LABELS
# ─────────────────────────────────────────────

# 26 Letters + "?" as the unknown/REST gesture
ascii_string = string.ascii_lowercase.upper() + "?"
labels_dict  = {idx: value for idx, value in enumerate(ascii_string)}

# Number labels 0–9 + "?" as unknown
number_labels_dict = {idx: str(idx) for idx in range(10)}
number_labels_dict[10] = "?"

# Colors in BGR format (OpenCV uses BGR, not RGB!)
BLACK  = (0,   0,   0  )
RED    = (0,   0,   255)
GREEN  = (0,   255, 0  )
BLUE   = (255, 0,   0  )
YELLOW = (0,   255, 255)
WHITE  = (255, 255, 255)
ORANGE = (0,   165, 255)
PURPLE = (255, 0,   200)


# ─────────────────────────────────────────────
# 2. MODEL HELPERS
# ─────────────────────────────────────────────

def load_model(model_path):
    """
    Load a trained Random Forest model from a .p pickle file.

    The pickle file is a dict with key 'model' — this matches
    how train.ipynb saves it:
        pickle.dump({'model': clf}, f)

    Usage:
        letter_model = load_model('./classifier/classify_letter_model.p')
        number_model = load_model('./classifier/classify_number_model.p')
    """
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict['model']


# ─────────────────────────────────────────────
# 3. DRAWING HELPERS  
# ─────────────────────────────────────────────

def calc_landmark_list(image, landmarks):
    """
    Convert MediaPipe's normalized coords (0.0–1.0) to pixel coords.
    Returns list of [x_px, y_px] for each of the 21 landmarks.
    Called every frame before draw_landmarks().
    """
    h, w = image.shape[:2]
    points = []
    for lm in landmarks.landmark:
        x = min(int(lm.x * w), w - 1)
        y = min(int(lm.y * h), h - 1)
        points.append([x, y])
    return points


def draw_landmarks(image, landmark_point):
    """
    Draw the full hand skeleton on the camera frame.
    Each bone: thick black outline + thinner white fill = clean look on any bg.
    Fingertips (4, 8, 12, 16, 20) drawn larger (8px) vs joints (5px).
    """
    if len(landmark_point) == 0:
        return image

    # All bone connections
    connections = [
        (2,3),(3,4),                        # thumb
        (5,6),(6,7),(7,8),                  # index
        (9,10),(10,11),(11,12),             # middle
        (13,14),(14,15),(15,16),            # ring
        (17,18),(18,19),(19,20),            # pinky
        (0,1),(1,2),(2,5),(5,9),            # palm
        (9,13),(13,17),(17,0),
    ]
    for a, b in connections:
        cv2.line(image, tuple(landmark_point[a]), tuple(landmark_point[b]), BLACK, 6)
        cv2.line(image, tuple(landmark_point[a]), tuple(landmark_point[b]), WHITE, 2)

    # Joint dots — fingertips are bigger
    fingertips = {4, 8, 12, 16, 20}
    for idx, pt in enumerate(landmark_point):
        r = 8 if idx in fingertips else 5
        cv2.circle(image, (pt[0], pt[1]), r, WHITE, -1)
        cv2.circle(image, (pt[0], pt[1]), r, BLACK,  1)

    return image


def draw_info_text(image, pos, hand_sign_text):
    """
    Draw a black label box showing the current predicted letter/number.
    pos = (x1, y1, x2) from the hand bounding box.
    """
    cv2.rectangle(
        image,
        (pos[0] - 2, pos[1]),
        (pos[2] + 2, pos[1] - 20),
        BLACK, -1
    )
    info_text = f'Label: {hand_sign_text}' if hand_sign_text else ''
    cv2.putText(
        image, info_text,
        (pos[0] + 5, pos[1] - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA
    )
    return image


def classify_landmark(landmark):
    """
    Split 21-point landmark list into anatomical finger groups.
    Returns: [wrist, thumb(4pts), index(4), middle(4), ring(4), pinky(4)]
    Used by is_finger_on() to check individual finger states.
    """
    return [
        [landmark[0]],   # wrist
        landmark[1:5],   # thumb
        landmark[5:9],   # index
        landmark[9:13],  # middle
        landmark[13:17], # ring
        landmark[17:21], # pinky
    ]


def is_finger_on(idx, finger, landmark_label):
    """
    Detect whether a finger is extended (up) or curled (down).
    idx=0 is thumb (extends sideways, use x-axis).
    idx=1–4 are other fingers (extend upward, use y-axis — lower y = higher).
    landmark_label: "Left" or "Right" (thumb direction differs per hand).
    """
    if idx == 0:
        if landmark_label == "Right":
            return finger[-1].x < finger[-2].x
        else:
            return finger[-1].x > finger[-2].x
    return finger[-1].y < finger[0].y


# ─────────────────────────────────────────────
# 4. SAVE HELPERS 
# ─────────────────────────────────────────────

def save_gif(gif_array, fps=30, output_dir="./assets/result.gif"):
    """Save a list of RGB frames as an animated GIF."""
    imageio.mimwrite(output_dir, gif_array, duration=1000/fps, format='GIF', loop=0)
    print(f"Saved GIF -> {output_dir}")


def save_video(frame_array, width, height, fps=30, output_dir="./assets/result.mp4"):
    """Save a list of BGR frames as an MP4 video."""
    writer = cv2.VideoWriter(
        output_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
    )
    for frame in frame_array:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.release()
    print(f"Saved video -> {output_dir}")


# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING  — NEW
# ─────────────────────────────────────────────
def extract_landmarks_from_image(img_path, hands):
    """
    Run MediaPipe on a saved IMAGE FILE and return 42-float feature vector.

    Different from extract_landmarks() which works on live MediaPipe results.
    This one takes an image PATH and handles extraction only:
        read image → run MediaPipe → extract raw [x0,y0,...,x20,y20]

    Used by:
        train.py      → extract landmarks from training images
        test_model.py → extract landmarks from test images

    Args:
        img_path : path to image file (.jpg, .png etc)
        hands    : initialized MediaPipe Hands object

    Returns:
        list of 42 floats, or None if no hand detected
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    # MediaPipe needs RGB, OpenCV loads BGR
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x, lm.y])

    return coords  # 42 raw floats

def extract_landmarks(results):
    """
    Pull the 21 (x, y) points from MediaPipe's results for the FIRST
    detected hand -> flat list of 42 floats.

    Returns None if no hand detected — use this in main.py to trigger
    space/pause detection (no hand = user put hand down = word boundary).

    Why flat? scikit-learn expects a 1D feature vector per sample.
    """
    if not results.multi_hand_landmarks:
        return None
    coords = []
    for lm in results.multi_hand_landmarks[0].landmark:
        coords.extend([lm.x, lm.y])
    return coords  # 21 × 2 = 42 values


def extract_landmarks_by_index(results, hand_index):
    """
    Same as extract_landmarks() but for a specific hand index.
    Use when two hands are visible — call once with 0, once with 1.
    Returns None if that hand index doesn't exist.
    """
    if not results.multi_hand_landmarks:
        return None
    if hand_index >= len(results.multi_hand_landmarks):
        return None
    coords = []
    for lm in results.multi_hand_landmarks[hand_index].landmark:
        coords.extend([lm.x, lm.y]) #x,y only no z
    return coords #42 values


def normalize_landmarks(landmarks):
    """
    Make the feature vector invariant to hand position and size.

        Problem without this:
            - Same sign on the left vs right of frame = different prediction
            - Same sign close vs far from camera = different prediction

            1. Subtract wrist (point 0) so wrist is always at (0, 0)
            2. Divide by wrist->middle_MCP distance (point 9) to normalize scale

    Returns 42 normalized floats.
    """
    pts = np.array(landmarks, dtype=np.float32).reshape(21, 2) #2d
    pts -= pts[0]                          # translate: wrist to origin
    scale = np.linalg.norm(pts[9])        # palm size = wrist -> middle base
    if scale < 1e-6:
        return [0.0] * 42                 # degenerate hand pose
    pts /= scale
    return pts.flatten().tolist()


def compute_finger_angles(landmarks):
    """
    Compute the bend angle at 15 finger joints (3 joints × 5 fingers).
    Adds 15 optional features on top of the 42 xy values.

    WHY THIS MATTERS:
    Signs A, S, and E all look like closed fists — their xyz coords
    are nearly identical. The joint angles are different though:
      A -> thumb tucked to side, fingers fully curled
      S -> thumb folded OVER fingers
      E -> fingers bent at knuckle but not fully closed
    These 15 numbers help the model split those hard cases apart.

    Each triplet (a, b, c) = angle measured AT joint b.
    Uses dot product: cos(θ) = (v1·v2) / (|v1||v2|)

    Returns 15 floats in range [0, π] radians.
    """
    pts = np.array(landmarks, dtype=np.float32).reshape(21, 2)

    # (point_a, vertex_b, point_c) — angle measured at b
    triplets = [
        (0,2,3),(2,3,4),(1,2,3),       # thumb
        (0,5,6),(5,6,7),(6,7,8),       # index
        (0,9,10),(9,10,11),(10,11,12), # middle
        (0,13,14),(13,14,15),(14,15,16),# ring
        (0,17,18),(17,18,19),(18,19,20),# pinky
    ]

    angles = []
    for a, b, c in triplets:
        v1 = pts[a] - pts[b]
        v2 = pts[c] - pts[b]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            angles.append(0.0)
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles.append(float(np.arccos(cos_a)))

    return angles  # 15 values


def build_feature_vector(landmarks, use_angles=False):
    """
    THE main function to call before model.predict().
    Build the model input feature vector.

    Args:
        landmarks:   raw 42-float list from extract_landmarks()
        use_angles:  False -> 42 features (matches train.py)
                     True  -> 57 features (42 + 15) for experimental models

    IMPORTANT:
      train.py saves models trained on 42 xy-only features.
      Keep use_angles=False for inference with those models.
    """
    normalized = normalize_landmarks(landmarks)
    if not use_angles:
        return normalized
    return normalized + compute_finger_angles(landmarks)  # 42 + 15 = 57


# ─────────────────────────────────────────────
# 6. HANDEDNESS  — NEW
# ─────────────────────────────────────────────

def get_handedness(results):
    """
    Read MediaPipe's built-in left/right hand detection.
    Returns dict: { hand_index → "Left" or "Right" }

    Ex:
        {0: "Right"}             — one right hand visible
        {0: "Left", 1: "Right"} — both hands visible

    Used in main.py to auto-route without the user pressing M:
        "Right" -> letter model
        "Left"  -> number model

    NOTE: MediaPipe labels are from the camera's perspective.
    If you flip the frame with cv2.flip(frame, 1) before processing
    (standard for selfie/mirror webcam), Left/Right will be correct
    from the user's perspective too.
    """
    result = {}
    if not results.multi_handedness:
        return result
    for i, hand_info in enumerate(results.multi_handedness):
        result[i] = hand_info.classification[0].label
    return result


def count_hands(results):
    """
    Returns 0, 1, or 2 — how many hands MediaPipe currently sees.
    Use in main.py to quickly decide single vs dual-hand processing path.
    """
    if not results.multi_hand_landmarks:
        return 0
    return len(results.multi_hand_landmarks)


def get_wrist_x(results, hand_index):
    """
    Returns the x-position (0.0–1.0) of a hand's wrist landmark.
    Used to sort two hands left-to-right so output order is always
    consistent regardless of which hand MediaPipe detects first.
    """
    if not results.multi_hand_landmarks:
        return 0.5
    if hand_index >= len(results.multi_hand_landmarks):
        return 0.5
    return results.multi_hand_landmarks[hand_index].landmark[0].x


# ─────────────────────────────────────────────
#MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Running utils.py self-test with dummy data...\n")
    np.random.seed(42)
    dummy = np.random.rand(42).tolist()

    norm = normalize_landmarks(dummy)
    assert len(norm) == 42, "Expected 42 values"
    print(f"  normalize_landmarks    : OK — 42 values, wrist ≈ {[round(v,4) for v in norm[:3]]}")

    angles = compute_finger_angles(dummy)
    assert len(angles) == 15, "Expected 15 angles"
    assert all(0 <= a <= np.pi for a in angles), "Angles must be in [0, π]"
    print(f"  compute_finger_angles  : OK — 15 values, range [{min(angles):.2f}, {max(angles):.2f}] rad")

    fv = build_feature_vector(dummy, use_angles=True)
    assert len(fv) == 57, "Expected 57 features"
    print(f"  build_feature_vector   : OK — {len(fv)} features (42 landmarks + 15 angles)")

    fv2 = build_feature_vector(dummy, use_angles=False)
    assert len(fv2) == 42, "Expected 42 features"
    print(f"  build_feature_vector   : OK — {len(fv2)} features (xy landmarks only)")

    print("\nAll tests passed,  utils.py is ready to use.")