import cv2
import mediapipe as mp
import numpy as np
import platform
from screeninfo import get_monitors
from scipy.ndimage import gaussian_filter

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
        "You will see red dots appear across the screen.",
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

    # === Define 3x3 Grid Calibration Points ===
    margin = 0.02  # Larger margin for better coverage
    xs = [margin, 0.5, 1 - margin]
    ys = [margin, 0.5, 1 - margin]
    calibration_targets = [(int(SCREEN_W * x), int(SCREEN_H * y)) for y in ys for x in xs]

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
            cv2.circle(calib_img, point, 12, (0, 0, 255), -1)
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

    # Use homography if we have >4 points
    if len(raw_points) >= 4:
        transform_matrix, _ = cv2.findHomography(raw_points, calib_points)
    else:
        transform_matrix = cv2.getPerspectiveTransform(raw_points[:4], calib_points[:4])

    cv2.destroyWindow("Calibration")
    return transform_matrix, baseline_distance

def create_heatmap(shape, points, sigma=30):
    """Create a heatmap from gaze points."""
    heatmap = np.zeros(shape[:2], dtype=np.float32)
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            heatmap[y, x] += 1
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def apply_heatmap_overlay(image, heatmap, alpha=0.5):
    """Apply heatmap overlay to the image."""
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access camera")
        return

    transform_matrix, baseline_distance = calibrate(cap)
    print("\nCalibration complete. Tracking started...")
    print("Press SPACE to finish viewing and see results.\n")

    # Load and resize product layout image
    item_page = cv2.imread("./product_page.png")
    if item_page is None:
        print("Error: Failed to load product image. Check file name and path.")
        return
    item_page = cv2.resize(item_page, (SCREEN_W, SCREEN_H))

    smoothing_factor = 0.2
    smoothed_mapped = None
    
    # Initialize gaze points collection
    gaze_points = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        display_frame = item_page.copy()

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_center = get_iris_center(face_landmarks, LEFT_IRIS_IDX, frame.shape)
            right_center = get_iris_center(face_landmarks, RIGHT_IRIS_IDX, frame.shape)
            gaze_point = compute_gaze_position(left_center, right_center)

            current_distance = np.linalg.norm(left_center - right_center)
            distance_scale = baseline_distance / current_distance
            adjusted_gaze = gaze_point * distance_scale

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

            # Store gaze point for final heatmap
            gaze_points.append(smoothed_mapped)

            # Draw gaze point
            cv2.circle(display_frame, (int(smoothed_mapped[0]), int(smoothed_mapped[1])), 10, (0, 255, 0), -1)

        cv2.imshow("Tracking", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE key
            print("\nSpace pressed — ending tracking.\n")
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    # Show final heatmap
    if gaze_points:
        print("\nGenerating heatmap visualization...")
        final_heatmap = create_heatmap(item_page.shape, gaze_points)
        final_overlay = apply_heatmap_overlay(item_page, final_heatmap)
        cv2.imshow("Final Heatmap", final_overlay)
        print("Press any key to close the heatmap window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
