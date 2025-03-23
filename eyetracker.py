import os
import time
import cv2
import numpy as np
import dlib


class EyeTrackingHeatmap:
    def __init__(self, stimulus_path=None, camera_id=0, heatmap_resolution=(800, 600)):
        self.camera_id = camera_id
        self.cap = None
        
        # Load Dlib face detector & shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.heatmap_resolution = heatmap_resolution
        self.stimulus_path = stimulus_path
        self.stimulus = self.load_stimulus()
        
        self.gaze_points = []
        self.heatmap = np.zeros((heatmap_resolution[1], heatmap_resolution[0]), dtype=np.float32)
        self.running = False
        
        # Calibration Data
        self.calibrated = False
        self.screen_corners = np.array([
            [0, 0],
            [heatmap_resolution[0], 0],
            [0, heatmap_resolution[1]],
            [heatmap_resolution[0], heatmap_resolution[1]],
        ])
        self.eye_corners = []

    def load_stimulus(self):
        if self.stimulus_path and os.path.exists(self.stimulus_path):
            stimulus = cv2.imread(self.stimulus_path)
            return cv2.resize(stimulus, self.heatmap_resolution)
        return np.ones((self.heatmap_resolution[1], self.heatmap_resolution[0], 3), dtype=np.uint8) * 255

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception("Could not open camera.")
    
    def calibrate(self):
        """Calibrate eye tracking by mapping eye positions to screen corners"""
        self.eye_corners = []
        corners_text = ["top-left", "top-right", "bottom-left", "bottom-right"]
        calibration_display = self.stimulus.copy()

        for i, corner in enumerate(self.screen_corners):
            cv2.circle(calibration_display, (int(corner[0]), int(corner[1])), 25, (0, 0, 255), -1)
            cv2.putText(
                calibration_display, f"Look at the {corners_text[i]} corner",
                (int(self.heatmap_resolution[0] / 3), int(self.heatmap_resolution[1] / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )

            cv2.imshow("Calibration", calibration_display)
            cv2.waitKey(500)

            eye_pos = None
            start_time = time.time()
            while time.time() - start_time < 3:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                eye_pos, _ = self.detect_eyes(frame)  # Get eye position

                if eye_pos:
                    cv2.circle(frame, (eye_pos[0], eye_pos[1]), 10, (0, 255, 0), -1)
                cv2.imshow("Eye Tracking", frame)
                cv2.waitKey(1)

            if eye_pos is not None:
                self.eye_corners.append(eye_pos)
            else:
                print(f"Could not detect eyes for {corners_text[i]} corner. Retrying...")
                i -= 1
                continue

            calibration_display = self.stimulus.copy()

        if len(self.eye_corners) == 4:
            self.eye_corners = np.array(self.eye_corners)
            self.calibrated = True
            cv2.destroyWindow("Calibration")
            print("Calibration complete!")
            return True

        print("Calibration failed. Please try again.")
        return False

    def detect_eyes(self, frame):
        """Detects eyes using Dlib’s shape predictor and returns eye position + outlines"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        for face in faces:
            landmarks = self.predictor(gray, face)

            # Get eye landmarks
            left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

            # Draw eye outlines
            cv2.polylines(frame, [left_eye_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(frame, [right_eye_pts], isClosed=True, color=(0, 255, 0), thickness=2)

            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            return eye_center, frame
        return None, frame

    def map_eye_to_screen(self, eye_pos):
        """Maps detected eye position to screen coordinates using homography"""
        if not self.calibrated or len(self.eye_corners) != 4:
            return None

        h, _ = cv2.findHomography(self.eye_corners, self.screen_corners)
        screen_pos = cv2.perspectiveTransform(np.array([eye_pos], dtype=np.float32).reshape(-1, 1, 2), h)
        return (int(screen_pos[0][0][0]), int(screen_pos[0][0][1]))

    def track_eyes(self, duration=10):
        """Tracks eye movement and updates heatmap"""
        self.start_camera()

        if not self.calibrated:
            if not self.calibrate():
                return False

        self.gaze_points = []
        self.heatmap = np.zeros((self.heatmap_resolution[1], self.heatmap_resolution[0]), dtype=np.float32)

        cv2.namedWindow("Stimulus")
        cv2.imshow("Stimulus", self.stimulus)

        start_time = time.time()
        print(f"Starting eye tracking for {duration} seconds...")

        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            eye_pos, frame = self.detect_eyes(frame)

            if eye_pos:
                screen_pos = self.map_eye_to_screen(eye_pos)
                if screen_pos and 0 <= screen_pos[0] < self.heatmap_resolution[0] and 0 <= screen_pos[1] < self.heatmap_resolution[1]:
                    self.gaze_points.append(screen_pos)
                    self._add_gaussian_to_heatmap(screen_pos)
                    cv2.circle(frame, eye_pos, 10, (0, 255, 0), -1)

                    # NEW: draw red dot at estimated screen position
                    stimulus_display = self.stimulus.copy()
                    cv2.circle(stimulus_display, screen_pos, 10, (0, 0, 255), -1)
                    cv2.imshow("Stimulus", stimulus_display)

            cv2.imshow("Eye Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        print(f"Eye tracking complete! Collected {len(self.gaze_points)} gaze points.")
        self.cap.release()
        cv2.destroyAllWindows()
        return True

    def _add_gaussian_to_heatmap(self, center, sigma=50):
        """Adds a Gaussian blob to heatmap at given center"""
        y, x = np.meshgrid(np.arange(self.heatmap_resolution[1]), np.arange(self.heatmap_resolution[0]), indexing="ij")
        dst = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        gaussian = np.exp(-(dst**2) / (2 * sigma**2))
        self.heatmap += gaussian

    def generate_heatmap(self, save_path="heatmap.png"):
        """Generates and displays gaze heatmap"""
        if len(self.gaze_points) == 0:
            print("No gaze data collected.")
            return
        
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(self.stimulus, 0.6, heatmap_color, 0.4, 0)

        cv2.imshow("Gaze Heatmap", blended)
        cv2.waitKey(0)
        cv2.imwrite(save_path, blended)
        print(f"Heatmap saved to {save_path}")


def main():
    tracker = EyeTrackingHeatmap()
    tracker.track_eyes(duration=10)
    tracker.generate_heatmap()

if __name__ == "__main__":
    main()
