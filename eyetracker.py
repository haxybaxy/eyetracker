import cv2
import mediapipe as mp
import numpy as np
import time
import platform
# Initialize Mediapipe Face Mesh with iris tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Iris landmark indices
LEFT_IRIS_IDX = [468, 469, 470, 471]
RIGHT_IRIS_IDX = [473, 474, 475, 476]

def get_screen_resolution():
    window_name = "temp_win"
    
    if platform.system() == "Darwin":
        from screeninfo import get_monitors
        monitor = get_monitors()[0]  # Get primary monitor
        return monitor.width * 2, monitor.height * 2 # for retina display
    else:
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_w = cv2.getWindowImageRect(window_name)[2]
    screen_h = cv2.getWindowImageRect(window_name)[3]
    cv2.destroyWindow(window_name)
    return screen_w, screen_h

SCREEN_W, SCREEN_H = get_screen_resolution()

def get_iris_center(landmarks, indices, frame_shape):
    h, w = frame_shape[:2]
    pts = []
    for i in indices:
        lm = landmarks.landmark[i]
        pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    return center

def compute_gaze_position(left_center, right_center):
    return (left_center + right_center) / 2

def calibrate(cap):
    # === Show Instructions Window ===
    instruction_text = [
        "Welcome to the first stage of this program: Calibration.",
        "",
        "You will see red dots appear in different corners of the screen.",
        "Please look directly at the red dot and press the spacebar.",
        "",
        "Once done, the next red dot will appear.",
        "",
        "Press any key to begin."
    ]

    instruction_img = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    line_height = 40
    start_y = SCREEN_H // 4

    for i, line in enumerate(instruction_text):
        size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        x = (SCREEN_W - size[0]) // 2
        y = start_y + i * line_height
        cv2.putText(instruction_img, line, (x, y), font, font_scale, (255, 255, 255), thickness)

    cv2.imshow("Calibration", instruction_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Calibration")

    # === Start Calibration Phase ===
    margin = 0.02
    calibration_targets = [
        (int(SCREEN_W * margin), int(SCREEN_H * margin)),                  # Top-left
        (int(SCREEN_W * (1 - margin)), int(SCREEN_H * margin)),            # Top-right
        (int(SCREEN_W * (1 - margin)), int(SCREEN_H * (1 - margin))),      # Bottom-right
        (int(SCREEN_W * margin), int(SCREEN_H * (1 - margin)))             # Bottom-left
    ]

    raw_points = []
    intereye_distances = []

    print("=== Calibration Mode ===")
    print("Look at each RED dot and press SPACE to capture.\n")

    for point in calibration_targets:
        captured = False
        while not captured:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            calib_img = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(calib_img, point, 12, (0, 0, 255), -1)  # Red dot
            cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Calibration", calib_img)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                left_center = get_iris_center(face_landmarks, LEFT_IRIS_IDX, frame.shape)
                right_center = get_iris_center(face_landmarks, RIGHT_IRIS_IDX, frame.shape)
                gaze_point = compute_gaze_position(left_center, right_center)
                cv2.circle(frame, (int(gaze_point[0]), int(gaze_point[1])), 5, (0, 255, 0), -1)

            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32 and results.multi_face_landmarks:  # SPACE
                raw_points.append(gaze_point)
                d = np.linalg.norm(left_center - right_center)
                intereye_distances.append(d)
                print(f"Captured gaze for {point}")
                captured = True

    baseline_distance = np.mean(intereye_distances)
    raw_points = np.array(raw_points, dtype=np.float32)
    calib_points = np.array(calibration_targets, dtype=np.float32)
    transform_matrix = cv2.getPerspectiveTransform(raw_points, calib_points)

    cv2.destroyWindow("Calibration")
    return transform_matrix, baseline_distance

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access camera")
        return

    transform_matrix, baseline_distance = calibrate(cap)
    print("\nCalibration complete. Tracking started...")
    print("Press 'Q' to exit.\n")

    # === Live Tracking on black screen ===
    smoothing_factor = 0.2
    smoothed_mapped = None
    start_time = time.time()
    duration = 30  # seconds

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        black_screen = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_center = get_iris_center(face_landmarks, LEFT_IRIS_IDX, frame.shape)
            right_center = get_iris_center(face_landmarks, RIGHT_IRIS_IDX, frame.shape)
            gaze_point = compute_gaze_position(left_center, right_center)

            current_distance = np.linalg.norm(left_center - right_center)
            distance_scale = baseline_distance / current_distance
            adjusted_gaze = (gaze_point - gaze_point * 0) * distance_scale

            raw_point = np.array([[[gaze_point[0], gaze_point[1]]]], dtype=np.float32)
            mapped_point = cv2.perspectiveTransform(raw_point, transform_matrix)
            mapped_point = mapped_point[0][0].astype(int)

            mapped_x = max(0, min(SCREEN_W - 1, mapped_point[0]))
            mapped_y = max(0, min(SCREEN_H - 1, mapped_point[1]))

            if smoothed_mapped is None:
                smoothed_mapped = np.array([mapped_x, mapped_y], dtype=np.float32)
            else:
                smoothed_mapped = smoothing_factor * np.array([mapped_x, mapped_y], dtype=np.float32) + \
                                  (1 - smoothing_factor) * smoothed_mapped

            cv2.circle(black_screen, (int(smoothed_mapped[0]), int(smoothed_mapped[1])), 10, (0, 255, 0), -1)

        cv2.imshow("Tracking", black_screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()
