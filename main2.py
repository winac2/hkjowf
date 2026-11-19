
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# ==========================================
# CONFIGURATION
# ==========================================


# mediapipe cũ dùng .solutions từ .30 trở xuống

MODEL_PATH = "./hand_gesture_model_main_DrHeHoIcTh.pkl"

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ==========================================
# LOAD ML MODEL
# ==========================================

print("[INFO] Loading ML model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("[INFO] Model loaded successfully!")

# Check expected features
if model.n_features_in_ != 42:
    raise ValueError(
        f"Model expects {model.n_features_in_} features, not 42!"
    )

# ==========================================
# INITIALIZE MEDIAPIPE
# ==========================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ==========================================
# LANDMARK EXTRACTION
# ==========================================

def extract_landmarks(hand_landmarks):

    landmarks = []

    for point in hand_landmarks.landmark:
        landmarks.extend([
            point.x,
            point.y
        ])

    # Convert to NumPy array
    X = np.array(
        landmarks,
        dtype=np.float32
    ).reshape(1, -1)

    return X


# ==========================================
# CAMERA INITIALIZATION
# ==========================================

print("[INFO] Starting camera...")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("[INFO] Camera started!")
print("[INFO] Press Q to quit.")

# ==========================================
# MAIN LOOP
# ==========================================

previous_time = time.time()

try:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("[WARNING] Cannot read camera frame!")
            break

        # Mirror frame
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Process hand landmarks
        results = hands.process(rgb_frame)

        prediction_text = "No hand detected"

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                # Extract 21 landmarks -> 42 features
                X = extract_landmarks(hand_landmarks)

                # ML prediction
                prediction = model.predict(X)[0]

                prediction_text = str(prediction)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # ==================================
        # FPS
        # ==================================

        current_time = time.time()

        fps = 1.0 / max(
            current_time - previous_time,
            0.000001
        )

        previous_time = current_time

        # ==================================
        # DISPLAY INFORMATION
        # ==================================

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 85),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            f"Gesture: {prediction_text}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.imshow(
            "Sign Machine - Raspberry Pi Test",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    # ======================================
    # CLEANUP
    # ======================================

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print("[INFO] Application closed.")
