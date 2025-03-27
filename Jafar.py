import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression

# === Setup ===
LEFT_IRIS_IDX = list(range(474, 478))
RIGHT_IRIS_IDX = list(range(469, 473))
BUFFER_SIZE = 5

# === Helper Functions ===

def get_screen_resolution():
    """Get the screen resolution dynamically."""
    window_name = "temp_window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_width = cv2.getWindowImageRect(window_name)[2]
    screen_height = cv2.getWindowImageRect(window_name)[3]
    cv2.destroyWindow(window_name)
    return screen_width, screen_height

# Dynamically get screen resolution
SCREEN_W, SCREEN_H = get_screen_resolution()

# Init MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Buffers for smoothing
left_iris_buffer = deque(maxlen=BUFFER_SIZE)
right_iris_buffer = deque(maxlen=BUFFER_SIZE)

def get_center(landmarks):
    x = np.mean([pt.x for pt in landmarks])
    y = np.mean([pt.y for pt in landmarks])
    return np.array([x, y])

def get_smoothed_iris_center(face_landmarks, w, h):
    left_raw = [face_landmarks.landmark[i] for i in LEFT_IRIS_IDX]
    right_raw = [face_landmarks.landmark[i] for i in RIGHT_IRIS_IDX]

    left_center = get_center(left_raw)
    right_center = get_center(right_raw)

    left_iris_buffer.append(left_center)
    right_iris_buffer.append(right_center)

    left_smooth = np.mean(left_iris_buffer, axis=0)
    right_smooth = np.mean(right_iris_buffer, axis=0)

    avg = (left_smooth + right_smooth) / 2
    return avg * [w, h]  # return pixel coords

# === Calibration Phase ===
def calibrate(cap):
    """Calibrate the gaze tracking system."""
    x_positions = np.linspace(0.02, 0.98, 5)
    y_positions = np.linspace(0.02, 0.98, 5)

    dot_positions = [
        (int(SCREEN_W * x), int(SCREEN_H * y))
        for y in y_positions
        for x in x_positions
    ]

    iris_data = []
    screen_data = []

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Calibration: Look at the dot and press SPACE")

    for i, (dot_x, dot_y) in enumerate(dot_positions):
        print(f"Point {i+1}/{len(dot_positions)}")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)  # Flip camera horizontally

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # Resize webcam feed to full screen
            resized_frame = cv2.resize(frame, (SCREEN_W, SCREEN_H))

            # Show calibration dot on top
            cv2.circle(resized_frame, (dot_x, dot_y), 10, (0, 0, 255), -1)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                iris_center = get_smoothed_iris_center(face_landmarks, w, h)
                cx, cy = iris_center.astype(int)

                # Scale iris center to screen resolution
                scaled_cx = int(cx * SCREEN_W / w)
                scaled_cy = int(cy * SCREEN_H / h)

                cv2.circle(resized_frame, (scaled_cx, scaled_cy), 3, (0, 255, 0), -1)
                cv2.putText(resized_frame, f"Center: ({scaled_cx}, {scaled_cy})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.imshow("Calibration", resized_frame)
                key = cv2.waitKey(1)
                if key == 32:  # SPACE
                    iris_data.append(iris_center)
                    screen_data.append([dot_x, dot_y])
                    break
            else:
                cv2.imshow("Calibration", resized_frame)
                cv2.waitKey(1)

    cv2.destroyWindow("Calibration")
    print("Calibration complete!")

    model_x = LinearRegression().fit(iris_data, [pt[0] for pt in screen_data])
    model_y = LinearRegression().fit(iris_data, [pt[1] for pt in screen_data])
    return model_x, model_y

# === Tracking Phase ===
def track_gaze(cap, model_x, model_y):
    """Track the user's gaze in real-time."""
    print("Gaze tracking... ESC to exit")

    cv2.namedWindow("Gaze Dot", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gaze Dot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # Flip camera horizontally
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Black background for tracking
        display_frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            iris_center = get_smoothed_iris_center(face_landmarks, w, h)
            pred_x = int(model_x.predict([iris_center])[0])
            pred_y = int(model_y.predict([iris_center])[0])

            cv2.circle(display_frame, (pred_x, pred_y), 10, (0, 0, 255), -1)

            cx, cy = iris_center.astype(int)

            # Scale iris center to screen resolution
            scaled_cx = int(cx * SCREEN_W / w)
            scaled_cy = int(cy * SCREEN_H / h)

            cv2.circle(display_frame, (scaled_cx, scaled_cy), 3, (0, 255, 0), -1)
            cv2.putText(display_frame, f"Center: ({scaled_cx}, {scaled_cy})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Gaze Dot", display_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# === Main ===
if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Dynamically get screen resolution
    SCREEN_W, SCREEN_H = get_screen_resolution()

    model_x, model_y = calibrate(cap)
    track_gaze(cap, model_x, model_y)

    cap.release()
    cv2.destroyAllWindows()
