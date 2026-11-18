
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "./hand_gesture_model_main_DrHeHoIcTh.pkl
"

HAND_MODEL_PATH = "./hand_landmarker.task"

CAMERA_INDEX = 0

NUM_HANDS = 2

# ==========================================
# LOAD ML MODEL
# ==========================================

print("[INFO] Loading gesture model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("[INFO] Gesture model loaded!")

# ==========================================
# INITIALIZE MEDIAPIPE
# ==========================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=NUM_HANDS,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# LANDMARK PROCESSING
# ==========================================

def extract_landmarks(hand_landmarks):

    # Extract x, y from 21 landmarks
    landmarks = []

    for landmark in hand_landmarks:
        landmarks.extend([
            landmark.x,
            landmark.y
        ])

    landmarks = np.array(
        landmarks,
        dtype=np.float32
    ).reshape(21, 2)

    # Normalize relative to minimum x and y
    landmarks[:, 0] -= np.min(landmarks[:, 0])
    landmarks[:, 1] -= np.min(landmarks[:, 1])

    # Convert to 42 features
    return landmarks.flatten().reshape(1, -1)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

if not cap.isOpened():
    print("[ERROR] Cannot open camera!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Camera started!")
print("[INFO] Press Q to quit.")

# ==========================================
# MAIN LOOP
# ==========================================

with HandLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("[WARNING] Cannot read camera frame.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = time.monotonic_ns() // 1_000_000

        # Detect hands
        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        prediction_text = "No hand detected"

        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                # Extract landmarks
                X = extract_landmarks(hand_landmarks)

                # Check input dimensions
                if X.shape[1] != model.n_features_in_:
                    raise ValueError(
                        f"Expected {model.n_features_in_} features, "
                        f"got {X.shape[1]}"
                    )

                # ML prediction
                prediction = model.predict(X)[0]

                prediction_text = str(prediction)

                # Draw landmarks
                height, width, _ = frame.shape

                points = []

                for landmark in hand_landmarks:
                    px = int(landmark.x * width)
                    py = int(landmark.y * height)
                    points.append((px, py))

                # Draw hand connections
                connections = (
                    mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                )

                for connection in connections:
                    start = points[connection.start]
                    end = points[connection.end]

                    cv2.line(
                        frame,
                        start,
                        end,
                        (0, 255, 0),
                        2
                    )

                # Draw landmark points
                for point in points:
                    cv2.circle(
                        frame,
                        point,
                        4,
                        (0, 0, 255),
                        -1
                    )

        # Display prediction
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 60),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            f"Gesture: {prediction_text}",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Sign Machine - Raspberry Pi Test",
            frame
        )

        # Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("[INFO] Application closed.")
