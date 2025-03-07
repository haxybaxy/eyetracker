import os
import time

import cv2
import numpy as np


class EyeTrackingHeatmap:
    def __init__(self, stimulus_path=None, camera_id=0, heatmap_resolution=(800, 600)):
        """
        Initialize the eye tracking system

        Parameters:
        -----------
        stimulus_path : str
            Path to the image that will be shown during tracking
        camera_id : int
            Camera device ID (default is 0 for built-in webcam)
        heatmap_resolution : tuple
            Resolution of the heatmap (width, height)
        """
        self.camera_id = camera_id
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        # Stimulus setup
        self.heatmap_resolution = heatmap_resolution
        self.stimulus_path = stimulus_path
        self.stimulus = None
        if stimulus_path and os.path.exists(stimulus_path):
            self.stimulus = cv2.imread(stimulus_path)
            self.stimulus = cv2.resize(self.stimulus, heatmap_resolution)
        else:
            self.stimulus = (
                np.ones(
                    (heatmap_resolution[1], heatmap_resolution[0], 3), dtype=np.uint8
                )
                * 255
            )

        # Gaze data collection
        self.gaze_points = []
        # Important fix: Make sure heatmap has shape (height, width) to match numpy convention
        self.heatmap = np.zeros(
            (heatmap_resolution[1], heatmap_resolution[0]), dtype=np.float32
        )
        self.running = False

        # Calibration
        self.calibrated = False
        self.screen_corners = np.array(
            [
                [0, 0],
                [heatmap_resolution[0], 0],
                [0, heatmap_resolution[1]],
                [heatmap_resolution[0], heatmap_resolution[1]],
            ]
        )
        self.eye_corners = []

    def start_camera(self):
        """Start the camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception(
                "Could not open camera. Check if camera is connected and ID is correct."
            )
        return self.cap.isOpened()

    def calibrate(self):
        """
        Calibrate eye tracking by looking at each corner of the screen
        """
        self.eye_corners = []
        corners_text = ["top-left", "top-right", "bottom-left", "bottom-right"]
        calibration_display = self.stimulus.copy()

        for i, corner in enumerate(self.screen_corners):
            # Draw a target at the corner
            cv2.circle(
                calibration_display,
                (int(corner[0]), int(corner[1])),
                25,
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                calibration_display,
                f"Look at the {corners_text[i]} corner",
                (
                    int(self.heatmap_resolution[0] / 3),
                    int(self.heatmap_resolution[1] / 2),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

            # Show the calibration display
            cv2.imshow("Calibration", calibration_display)
            cv2.waitKey(500)  # Small delay

            # Collect eye position for this corner
            eye_pos = None
            start_time = time.time()
            while time.time() - start_time < 3:  # 3 seconds to look at each corner
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)  # Mirror display for intuitive movement
                faces = self.face_cascade.detectMultiScale(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5
                )

                for x, y, w, h in faces:
                    roi_gray = cv2.cvtColor(
                        frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY
                    )
                    eyes = self.eye_cascade.detectMultiScale(roi_gray)

                    if len(eyes) >= 2:
                        # Calculate the center point between the two eyes
                        eye_centers = []
                        for ex, ey, ew, eh in eyes[:2]:  # Use first two detected eyes
                            eye_center = [x + ex + ew // 2, y + ey + eh // 2]
                            eye_centers.append(eye_center)
                            cv2.circle(
                                frame,
                                (eye_center[0], eye_center[1]),
                                5,
                                (255, 0, 0),
                                -1,
                            )

                        # Average position of the two eyes
                        eye_pos = np.mean(eye_centers, axis=0)
                        cv2.circle(
                            frame,
                            (int(eye_pos[0]), int(eye_pos[1])),
                            10,
                            (0, 255, 0),
                            -1,
                        )

                # Display frame with eye tracking
                cv2.imshow("Eye Tracking", frame)
                cv2.waitKey(1)

            if eye_pos is not None:
                self.eye_corners.append(eye_pos)
            else:
                print(
                    f"Could not detect eyes for {corners_text[i]} corner. Retrying..."
                )
                i -= 1  # Retry this corner
                continue

            # Reset calibration display
            calibration_display = self.stimulus.copy()

        if len(self.eye_corners) == 4:
            self.eye_corners = np.array(self.eye_corners)
            self.calibrated = True
            cv2.destroyWindow("Calibration")
            print("Calibration complete!")
            return True
        print("Calibration failed. Please try again.")
        return False

    def map_eye_to_screen(self, eye_pos):
        """Map eye position to screen coordinates using perspective transform"""
        if not self.calibrated or len(self.eye_corners) != 4:
            return None

        # Calculate homography matrix
        h, status = cv2.findHomography(self.eye_corners, self.screen_corners)

        # Transform eye position to screen coordinates
        eye_pos_array = np.array([eye_pos], dtype=np.float32)
        screen_pos = cv2.perspectiveTransform(eye_pos_array.reshape(-1, 1, 2), h)

        return (int(screen_pos[0][0][0]), int(screen_pos[0][0][1]))

    def track_eyes(self, duration=10):
        """
        Track eyes for the specified duration and collect gaze points

        Parameters:
        -----------
        duration : int
            Duration in seconds to track eyes
        """
        if not self.cap or not self.cap.isOpened():
            if not self.start_camera():
                return False

        # First calibrate if not already done
        if not self.calibrated:
            if not self.calibrate():
                return False

        # Reset data for new tracking session
        self.gaze_points = []
        # Important fix: Make sure heatmap has shape (height, width) to match numpy convention
        self.heatmap = np.zeros(
            (self.heatmap_resolution[1], self.heatmap_resolution[0]), dtype=np.float32
        )

        # Set up display window
        cv2.namedWindow("Stimulus")
        cv2.imshow("Stimulus", self.stimulus)

        # Main tracking loop
        self.running = True
        start_time = time.time()
        print(f"Starting eye tracking for {duration} seconds...")

        while self.running and (time.time() - start_time < duration):
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)  # Mirror for more intuitive interaction
            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for x, y, w, h in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y : y + h, x : x + w]

                # Detect eyes
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) >= 2:
                    # Calculate the center point between eyes
                    eye_centers = []
                    for ex, ey, ew, eh in eyes[:2]:
                        eye_center = [x + ex + ew // 2, y + ey + eh // 2]
                        eye_centers.append(eye_center)
                        cv2.circle(
                            display_frame,
                            (eye_center[0], eye_center[1]),
                            5,
                            (0, 0, 255),
                            -1,
                        )

                    # Average position of the two eyes
                    eye_pos = np.mean(eye_centers, axis=0)
                    cv2.circle(
                        display_frame,
                        (int(eye_pos[0]), int(eye_pos[1])),
                        10,
                        (0, 255, 0),
                        -1,
                    )

                    # Map eye position to screen coordinates
                    screen_pos = self.map_eye_to_screen(eye_pos)
                    if screen_pos:
                        # Add to gaze points if within bounds
                        if (
                            0 <= screen_pos[0] < self.heatmap_resolution[0]
                            and 0 <= screen_pos[1] < self.heatmap_resolution[1]
                        ):
                            self.gaze_points.append(screen_pos)
                            # Update heatmap with Gaussian around gaze point
                            self._add_gaussian_to_heatmap(screen_pos)

            # Display remaining time
            remaining = int(duration - (time.time() - start_time))
            cv2.putText(
                display_frame,
                f"Time remaining: {remaining}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Display the tracking frame
            cv2.imshow("Eye Tracking", display_frame)

            # Check for exit key
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                self.running = False

        print(f"Eye tracking complete! Collected {len(self.gaze_points)} gaze points.")
        self.running = False
        return True

    def _add_gaussian_to_heatmap(self, center, sigma=50):
        """Add a Gaussian blob to the heatmap at the given center point"""
        # Important fix: Create meshgrid with correct dimensions (y, x) to match heatmap shape
        y, x = np.meshgrid(
            np.arange(self.heatmap_resolution[1]),
            np.arange(self.heatmap_resolution[0]),
            indexing="ij",
        )
        dst = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        gaussian = np.exp(-(dst**2) / (2 * sigma**2))
        self.heatmap += gaussian

    def generate_heatmap(self, save_path=None):
        """
        Generate and display the gaze heatmap

        Parameters:
        -----------
        save_path : str
            Path to save the heatmap image (optional)
        """
        if len(self.gaze_points) == 0:
            print("No gaze data collected. Run track_eyes() first.")
            return None

        # Normalize heatmap
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_normalized = heatmap_normalized.astype(np.uint8)

        # Apply color map
        heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # Blend with stimulus image
        alpha = 0.7
        blended = cv2.addWeighted(self.stimulus, 1 - alpha, heatmap_color, alpha, 0)

        # Display result
        cv2.imshow("Gaze Heatmap", blended)
        cv2.waitKey(0)

        # Save if requested
        if save_path:
            cv2.imwrite(save_path, blended)
            print(f"Heatmap saved to {save_path}")

        return blended

    def close(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    # Get stimulus image from user
    stimulus_path = input(
        "Enter path to stimulus image (leave empty to use blank screen): "
    ).strip()
    if stimulus_path and not os.path.exists(stimulus_path):
        print(f"Warning: File {stimulus_path} not found. Using blank screen.")
        stimulus_path = None

    # Create eye tracker with specified stimulus
    tracker = EyeTrackingHeatmap(stimulus_path=stimulus_path)

    try:
        # Start camera
        if not tracker.start_camera():
            print("Failed to start camera. Exiting.")
            return

        # Run calibration and tracking
        print(
            "First, let's calibrate by looking at each corner of the screen when prompted."
        )
        if tracker.calibrate():
            duration = int(input("Enter tracking duration in seconds: ") or "10")
            if tracker.track_eyes(duration):
                save_path = input(
                    "Enter path to save heatmap (or press Enter to skip saving): "
                ).strip()
                tracker.generate_heatmap(save_path if save_path else None)
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
