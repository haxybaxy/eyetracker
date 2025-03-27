import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from sklearn.linear_model import LinearRegression

# === Setup ===
LEFT_IRIS_IDX = list(range(474, 478))
RIGHT_IRIS_IDX = list(range(469, 473))
SCREEN_W = 640
SCREEN_H = 480
BUFFER_SIZE = 5

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

# === Helper Functions ===
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
    dot_positions = [
        (SCREEN_W * x, SCREEN_H * y)
        for y in [0.1, 0.3, 0.5, 0.7, 0.9]
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]
    ]

    iris_data = []
    screen_data = []

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", SCREEN_W, SCREEN_H)

    print("Calibration: Look at the dot and press SPACE")

    for i, (dot_x, dot_y) in enumerate(dot_positions):
        print(f"Point {i+1}/{len(dot_positions)}")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # Show calibration dot
            overlay = frame.copy()
            cv2.circle(overlay, (int(dot_x), int(dot_y)), 10, (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
                )

                iris_center = get_smoothed_iris_center(face_landmarks, w, h)
                cx, cy = iris_center.astype(int)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                cv2.putText(frame, f"Center: ({cx}, {cy})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.imshow("Calibration", frame)
                key = cv2.waitKey(1)
                if key == 32:  # SPACE
                    iris_data.append(iris_center)
                    screen_data.append([dot_x, dot_y])
                    break
            else:
                cv2.imshow("Calibration", frame)
                cv2.waitKey(1)

    cv2.destroyWindow("Calibration")
    print("Calibration complete!")

    model_x = LinearRegression().fit(iris_data, [pt[0] for pt in screen_data])
    model_y = LinearRegression().fit(iris_data, [pt[1] for pt in screen_data])
    return model_x, model_y

# === Tracking Phase ===
def track_gaze(cap, model_x, model_y):
    print("Gaze tracking... ESC to exit")
    cv2.namedWindow("Iris Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Iris Tracking", SCREEN_W, SCREEN_H)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        gaze_screen = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            )

            iris_center = get_smoothed_iris_center(face_landmarks, w, h)
            pred_x = int(model_x.predict([iris_center])[0])
            pred_y = int(model_y.predict([iris_center])[0])

            cv2.circle(gaze_screen, (pred_x, pred_y), 10, (0, 0, 255), -1)

            cx, cy = iris_center.astype(int)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
            cv2.putText(frame, f"Center: ({cx}, {cy})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Iris Tracking", frame)
        cv2.imshow("Gaze Dot", gaze_screen)

        if cv2.waitKey(1) & 0xFF == 27:
            break

# === Main ===
if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    model_x, model_y = calibrate(cap)
    track_gaze(cap, model_x, model_y)
    cap.release()
    cv2.destroyAllWindows()
