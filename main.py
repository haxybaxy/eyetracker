"""
Eye Tracker Application
This module provides functionality for eye tracking and gaze analysis.
It supports two modes: YouTube video analysis and product layout analysis.
"""

import cv2
import mediapipe as mp
import numpy as np
import platform
from screeninfo import get_monitors
from scipy.ndimage import gaussian_filter
from pytubefix import YouTube
import ssl
import os
import time

# MediaPipe Face Mesh Constants
LEFT_IRIS_IDX = [468, 469, 470, 471]  # Indices for left iris landmarks
RIGHT_IRIS_IDX = [473, 474, 475, 476]  # Indices for right iris landmarks

# Calibration Constants
CALIBRATION_MARGIN = 0.02  # Margin from screen edges for calibration points
CALIBRATION_GRID_SIZE = 3  # Size of the calibration grid (3x3)

# Gaze Tracking Constants
SMOOTHING_FACTOR = 0.2  # Factor for smoothing gaze point movements
FACE_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for face detection and tracking

# Visualization Constants
HEATMAP_SIGMA = 30  # Standard deviation for Gaussian blur in heatmap
HEATMAP_ALPHA = 0.5  # Transparency factor for heatmap overlay
CALIBRATION_DOT_RADIUS = 12  # Radius of calibration dots in pixels
GAZE_POINT_RADIUS = 10  # Radius of gaze point visualization in pixels

class EyeTracker:
    """Main class for eye tracking functionality.
    
    This class handles all eye tracking operations including:
    - Face mesh detection and landmark tracking
    - Gaze point calculation and smoothing
    - Calibration and coordinate mapping
    - Heatmap visualization
    - YouTube video and product layout analysis modes
    """
    
    def __init__(self):
        """Initialize the eye tracker with necessary components.
        
        Sets up:
        - MediaPipe face mesh for facial landmark detection
        - Screen resolution for coordinate mapping
        - Visualization parameters
        """
        # Initialize MediaPipe face mesh for facial landmark detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # Track only one face
            refine_landmarks=True,  # Enable refined landmark detection
            min_detection_confidence=FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_DETECTION_CONFIDENCE
        )
        # Get screen resolution for proper coordinate mapping
        self.screen_w, self.screen_h = self._get_screen_resolution()
        print(f"\nScreen Resolution: {self.screen_w}x{self.screen_h} pixels")
        
        
    def _get_screen_resolution(self):
        """Get the screen resolution based on the operating system."""
        if platform.system() == "Darwin":  # macOS
            monitor = get_monitors()[0]
            return monitor.width, monitor.height # Account for retina display
        
        # For other operating systems, use OpenCV to get screen resolution
        window_name = "temp_win"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        screen_w = cv2.getWindowImageRect(window_name)[2]
        screen_h = cv2.getWindowImageRect(window_name)[3]
        cv2.destroyWindow(window_name)
        return screen_w, screen_h

    def _get_iris_center(self, landmarks, indices, frame_shape):
        """Calculate the center point of the iris based on landmark points."""
        h, w = frame_shape[:2]
        pts = []
        for i in indices:
            lm = landmarks.landmark[i]
            pts.append([int(lm.x * w), int(lm.y * h)])
        pts = np.array(pts)
        return np.mean(pts, axis=0)

    def _compute_gaze_position(self, left_center, right_center):
        """Compute the gaze position as the average of left and right iris centers."""
        return (left_center + right_center) / 2

    def _create_heatmap(self, shape, points, sigma=HEATMAP_SIGMA):
        """Create a heatmap visualization from gaze points."""
        heatmap = np.zeros(shape[:2], dtype=np.float32)
        for point in points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                heatmap[y, x] += 1
        # Apply Gaussian blur for smooth visualization
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        # Normalize heatmap values
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    def _apply_heatmap_overlay(self, image, heatmap, alpha=HEATMAP_ALPHA):
        """Apply heatmap overlay to the image with specified transparency."""
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    def calibrate(self, cap):
        """Perform calibration for eye tracking.
        
        This method guides the user through a calibration process where they look at
        predefined points on the screen. It collects gaze data and computes the
        transformation matrix for accurate screen coordinate mapping.
        
        Args:
            cap: OpenCV video capture object for camera input
            
        Returns:
            tuple: (transform_matrix, baseline_distance)
                - transform_matrix: 3x3 homography matrix for coordinate mapping
                - baseline_distance: Average inter-eye distance during calibration
        """
        # Show instructions to the user
        self._show_calibration_instructions()
        
        # Define calibration points in a 3x3 grid
        xs = [CALIBRATION_MARGIN, 0.5, 1 - CALIBRATION_MARGIN]
        ys = [CALIBRATION_MARGIN, 0.5, 1 - CALIBRATION_MARGIN]
        calibration_targets = [(int(self.screen_w * x), int(self.screen_h * y)) 
                             for y in ys for x in xs]

        # Initialize data collection arrays
        raw_points = []  # Store raw gaze points during calibration
        intereye_distances = []  # Store inter-eye distances for baseline calculation

        print("=== Calibration Mode ===")
        print("Look at each RED dot and press SPACE to capture.\n")

        # Create calibration window
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration", self.screen_w, self.screen_h)

        # Collect calibration data for each point
        for point in calibration_targets:
            captured = False
            while not captured:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)  # Mirror the frame horizontally
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(frame_rgb)

                # Display calibration target
                calib_img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                cv2.circle(calib_img, point, CALIBRATION_DOT_RADIUS, (0, 0, 255), -1)
                cv2.imshow("Calibration", calib_img)

                # Process face landmarks and display gaze point
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    left_center = self._get_iris_center(face_landmarks, LEFT_IRIS_IDX, frame.shape)
                    right_center = self._get_iris_center(face_landmarks, RIGHT_IRIS_IDX, frame.shape)
                    gaze_point = self._compute_gaze_position(left_center, right_center)
                    cv2.circle(frame, (int(gaze_point[0]), int(gaze_point[1])), 5, (0, 255, 0), -1)

                key = cv2.waitKey(1) & 0xFF
                if key == 32 and results.multi_face_landmarks:  # SPACE key pressed
                    raw_points.append(gaze_point)
                    d = np.linalg.norm(left_center - right_center)
                    intereye_distances.append(d)
                    print(f"Captured gaze for {point}")
                    captured = True

        # Calculate baseline distance and transformation matrix
        baseline_distance = np.mean(intereye_distances)
        raw_points = np.array(raw_points, dtype=np.float32)
        calib_points = np.array(calibration_targets, dtype=np.float32)

        # Compute transformation matrix using homography
        if len(raw_points) >= 4:
            transform_matrix, _ = cv2.findHomography(raw_points, calib_points)
        else:
            transform_matrix = cv2.getPerspectiveTransform(raw_points[:4], calib_points[:4])

        return transform_matrix, baseline_distance

    def _show_calibration_instructions(self):
        """Display calibration instructions to the user."""
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

        # Create and display instruction screen
        instruction_img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        line_height = 40
        start_y = self.screen_h // 4

        for i, line in enumerate(instruction_text):
            size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            x = (self.screen_w - size[0]) // 2
            y = start_y + i * line_height
            cv2.putText(instruction_img, line, (x, y), font, font_scale, (255, 255, 255), thickness)

        # Create instruction window with specific size
        cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration", self.screen_w, self.screen_h)
        cv2.imshow("Calibration", instruction_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE key
            print("\nSpace pressed — closing calibration window.\n")
            cv2.destroyWindow("Calibration")

    def track_gaze(self, cap, transform_matrix, baseline_distance, display_frame, on_gaze_point=None):
        """Track gaze and update the display frame."""
        frame = cv2.flip(cap.read()[1], 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_center = self._get_iris_center(face_landmarks, LEFT_IRIS_IDX, frame.shape)
            right_center = self._get_iris_center(face_landmarks, RIGHT_IRIS_IDX, frame.shape)
            gaze_point = self._compute_gaze_position(left_center, right_center)

            # Adjust gaze point based on current inter-eye distance
            current_distance = np.linalg.norm(left_center - right_center)
            distance_scale = baseline_distance / current_distance
            adjusted_gaze = gaze_point * distance_scale

            # Map gaze point to screen coordinates
            raw_point = np.array([[[gaze_point[0], gaze_point[1]]]], dtype=np.float32)
            mapped_point = cv2.perspectiveTransform(raw_point, transform_matrix)
            mapped_point = mapped_point[0][0].astype(int)

            # Ensure mapped point is within screen bounds
            mapped_x = max(0, min(self.screen_w - 1, mapped_point[0]))
            mapped_y = max(0, min(self.screen_h - 1, mapped_point[1]))

            if on_gaze_point:
                on_gaze_point(mapped_x, mapped_y, display_frame)

        return display_frame

    def run_youtube_version(self, cap, transform_matrix, baseline_distance, custom_url=None):
        """Run the YouTube video analysis version."""
        try:
            # Download and prepare YouTube video
            if custom_url:
                youtube_url = custom_url
            else:
                youtube_url = "https://www.youtube.com/watch?v=dz_GbQ2YFN4"
            ssl._create_default_https_context = ssl._create_unverified_context
            
            # Add retry logic for video download
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    yt = YouTube(youtube_url)
                    stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').first()
                    if stream is None:
                        raise Exception("No suitable video stream found")
                    video_path = stream.download(filename='temp_video')
                    break
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)
            
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise Exception("Failed to open downloaded video")
                
        except Exception as e:
            print(f"Error loading YouTube video: {e}")
            print("Please check your internet connection and try again.")
            print("Alternatively, you can use a local video file instead.")
            return

        print("\nCalibration complete. Tracking started...")
        print("Press SPACE to finish viewing and see results.\n")

        smoothed_mapped = None
        gaze_points = []
        frame_gaze_points = []  # Store gaze points with their frame numbers

        def on_gaze_point(mapped_x, mapped_y, display_frame):
            nonlocal smoothed_mapped
            # Apply smoothing to gaze point
            if smoothed_mapped is None:
                smoothed_mapped = np.array([mapped_x, mapped_y], dtype=np.float32)
            else:
                smoothed_mapped = SMOOTHING_FACTOR * np.array([mapped_x, mapped_y], dtype=np.float32) + \
                                (1 - SMOOTHING_FACTOR) * smoothed_mapped

            gaze_points.append(smoothed_mapped)
            # Store frame number with gaze point
            current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
            frame_gaze_points.append((current_frame, smoothed_mapped))
            cv2.circle(display_frame, (int(smoothed_mapped[0]), int(smoothed_mapped[1])), 10, (0, 255, 0), -1)

        # Create windows with specific size
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("Tracking", self.screen_w, self.screen_h)

        # Main tracking loop
        while True:
            ret, frame = cap.read()
            ret_video, video_frame = video.read()
            
            if not ret or not ret_video:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                ret_video, video_frame = video.read()
                continue

            video_frame = cv2.resize(video_frame, (self.screen_w, self.screen_h))
            display_frame = self.track_gaze(cap, transform_matrix, baseline_distance, video_frame.copy(), on_gaze_point)

            cv2.imshow("Tracking", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE key
                print("\nSpace pressed — ending tracking.\n")
                break

        # Clean up video capture
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_last, last_frame = video.read()
        video.release()
        
        # After tracking loop ends, show replay
        if frame_gaze_points:
            print("\nShowing replay with gaze points...")
            # Create a new video capture for replay
            replay_video = cv2.VideoCapture(video_path)
            
            # Create replay window
            cv2.namedWindow("Replay", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Replay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow("Replay", self.screen_w, self.screen_h)

            current_gaze_idx = 0
            trail_length = 30  # Number of previous gaze points to show in trail

            while True:
                ret_video, video_frame = replay_video.read()
                if not ret_video:
                    break

                current_frame = replay_video.get(cv2.CAP_PROP_POS_FRAMES)
                display_frame = cv2.resize(video_frame, (self.screen_w, self.screen_h))

                # Draw gaze trail
                while (current_gaze_idx < len(frame_gaze_points) and 
                       frame_gaze_points[current_gaze_idx][0] <= current_frame):
                    current_gaze_idx += 1

                # Draw trail of recent gaze points with fading effect
                start_idx = max(0, current_gaze_idx - trail_length)
                for i, (_, gaze_point) in enumerate(frame_gaze_points[start_idx:current_gaze_idx]):
                    alpha = (i + 1) / trail_length  # Fade older points
                    color = (0, int(255 * alpha), 0)  # Green with fading intensity
                    cv2.circle(display_frame, 
                             (int(gaze_point[0]), int(gaze_point[1])), 
                             5, color, -1)

                cv2.imshow("Replay", display_frame)
                if cv2.waitKey(1) & 0xFF == 32:  # SPACE to skip replay
                    break

            # Clean up replay video
            replay_video.release()
            cv2.destroyWindow("Replay")

        # Show final heatmap
        if gaze_points and ret_last:
            print("\nGenerating heatmap visualization...")
            last_frame = cv2.resize(last_frame, (self.screen_w, self.screen_h))
            final_heatmap = self._create_heatmap(last_frame.shape, gaze_points)
            final_overlay = self._apply_heatmap_overlay(last_frame, final_heatmap)
            
            # Create heatmap window with specific size
            cv2.namedWindow("Final Heatmap", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Final Heatmap", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow("Final Heatmap", self.screen_w, self.screen_h)
            cv2.imshow("Final Heatmap", final_overlay)
            print("Press any key to close the heatmap window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Warning: Could not generate heatmap - failed to capture final frame")

        # Clean up temporary video file
        if os.path.exists('temp_video'):
            os.remove('temp_video')

    def run_product_version(self, cap, transform_matrix, baseline_distance):
        """Run the product layout analysis version."""
        print("\nCalibration complete. Tracking started...")
        print("Press SPACE to finish viewing and see results.\n")

        # Load and prepare product page image
        item_page = cv2.imread("./product_page.png")
        if item_page is None:
            print("Error: Failed to load product image. Check file name and path.")
            return
        item_page = cv2.resize(item_page, (self.screen_w, self.screen_h))

        # Create windows with specific size
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("Tracking", self.screen_w, self.screen_h)

        # Define product regions on the page
        half_width = self.screen_w // 2
        half_height = self.screen_h // 2
        products = {
            "perfume":    ((0, 0), (half_width, half_height)),
            "shoe":       ((half_width, 0), (self.screen_w, half_height)),
            "watch":      ((0, half_height), (half_width, self.screen_h)),
            "sunglasses": ((half_width, half_height), (self.screen_w, self.screen_h))
        }

        def check_gaze_region(gaze_point, product_boxes):
            """Check which product region the gaze point falls into."""
            x, y = int(gaze_point[0]), int(gaze_point[1])
            for product, ((x1, y1), (x2, y2)) in product_boxes.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return product
            return None

        attention_counter = {key: 0 for key in products}
        smoothed_mapped = None
        gaze_points = []
        frame_gaze_points = []  # Add this to store timestamped gaze points
        start_time = time.time()  # Add this to track timing

        def on_gaze_point(mapped_x, mapped_y, display_frame):
            nonlocal smoothed_mapped
            if smoothed_mapped is None:
                smoothed_mapped = np.array([mapped_x, mapped_y], dtype=np.float32)
            else:
                smoothed_mapped = SMOOTHING_FACTOR * np.array([mapped_x, mapped_y], dtype=np.float32) + \
                                (1 - SMOOTHING_FACTOR) * smoothed_mapped

            gaze_points.append(smoothed_mapped)
            # Store timestamp with gaze point
            current_time = time.time() - start_time
            frame_gaze_points.append((current_time, smoothed_mapped))

            # Track which product is being looked at
            region = check_gaze_region(smoothed_mapped, products)
            if region:
                attention_counter[region] += 1

            # Display gaze point and coordinates
            cv2.circle(display_frame, (int(smoothed_mapped[0]), int(smoothed_mapped[1])), 10, (0, 255, 0), -1)
            cv2.putText(display_frame,
                f"X: {int(smoothed_mapped[0])}, Y: {int(smoothed_mapped[1])}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

        # Main tracking loop
        while True:
            display_frame = self.track_gaze(cap, transform_matrix, baseline_distance, item_page.copy(), on_gaze_point)
            cv2.imshow("Tracking", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE key
                print("\nSpace pressed — ending tracking.\n")
                break

        # Show replay
        if frame_gaze_points:
            print("\nShowing replay with gaze points...")
            
            # Create replay window
            cv2.namedWindow("Replay", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Replay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow("Replay", self.screen_w, self.screen_h)

            # Calculate replay speed (compress viewing time to 5 seconds)
            total_time = frame_gaze_points[-1][0]  # Last timestamp
            time_scale = total_time / 5.0  # Scale time to show everything in 5 seconds
            
            current_gaze_idx = 0
            trail_length = 30  # Number of previous gaze points to show in trail
            replay_start_time = time.time()

            while True:
                display_frame = item_page.copy()
                current_time = (time.time() - replay_start_time) * time_scale

                # Draw gaze trail
                while (current_gaze_idx < len(frame_gaze_points) and 
                       frame_gaze_points[current_gaze_idx][0] <= current_time):
                    current_gaze_idx += 1

                # Draw trail of recent gaze points with fading effect
                start_idx = max(0, current_gaze_idx - trail_length)
                for i, (_, gaze_point) in enumerate(frame_gaze_points[start_idx:current_gaze_idx]):
                    alpha = (i + 1) / trail_length  # Fade older points
                    color = (0, int(255 * alpha), 0)  # Green with fading intensity
                    cv2.circle(display_frame, 
                             (int(gaze_point[0]), int(gaze_point[1])), 
                             5, color, -1)

                cv2.imshow("Replay", display_frame)
                if cv2.waitKey(1) & 0xFF == 32 or current_gaze_idx >= len(frame_gaze_points):  # SPACE or finished
                    break

            cv2.destroyWindow("Replay")

        # Generate and display final heatmap
        if gaze_points:
            print("\nGenerating heatmap visualization...")
            final_heatmap = self._create_heatmap(item_page.shape, gaze_points)
            final_overlay = self._apply_heatmap_overlay(item_page, final_heatmap)
            
            # Create heatmap window with specific size
            cv2.namedWindow("Final Heatmap", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Final Heatmap", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.resizeWindow("Final Heatmap", self.screen_w, self.screen_h)
            cv2.imshow("Final Heatmap", final_overlay)
            print("Press any key to close the heatmap window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Warning: Could not generate heatmap - no gaze points recorded")

        # Display attention analysis results
        if attention_counter:
            most_looked_at = max(attention_counter, key=attention_counter.get)
            print("\nYou looked most at:", most_looked_at)
            print("Full attention breakdown:")
            for product, count in attention_counter.items():
                print(f"{product}: {count} frames")
        else:
            print("\nNo gaze data was recorded for analysis.")

    def cleanup(self):
        """Clean up resources."""
        self.face_mesh.close()

def main():
    """Main entry point for the eye tracker application.
    
    This function:
    1. Displays welcome message and version options
    2. Gets custom URL if needed
    3. Initializes the eye tracker and camera
    4. Performs calibration
    5. Runs the selected version (YouTube, Product, or Custom YouTube)
    6. Handles cleanup and error cases
    """
    try:
        # Display welcome message and get user choice
        print("\nWelcome to the Eye Tracker Application!")
        print("Please choose which version you want to run:")
        print("1. YouTube Video Version (Default Video)")
        print("2. Product Layout Version")
        print("3. Custom YouTube Video Version")
        
        choice = get_user_choice()
        
        # Get custom URL if needed
        custom_url = None
        if choice == 3:
            custom_url = input("\nPlease enter the YouTube URL: ")
        
        # Initialize components
        eye_tracker = EyeTracker()
        cap = initialize_camera()
        
        if not cap:
            return
            
        try:
            # Perform calibration
            transform_matrix, baseline_distance = eye_tracker.calibrate(cap)
            
            # Run selected version
            if choice == 1:
                eye_tracker.run_youtube_version(cap, transform_matrix, baseline_distance)
            elif choice == 2:
                eye_tracker.run_product_version(cap, transform_matrix, baseline_distance)
            else:  # choice == 3
                eye_tracker.run_youtube_version(cap, transform_matrix, baseline_distance, custom_url=custom_url)
        finally:
            # Clean up resources
            cap.release()
            eye_tracker.cleanup()
            
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check your camera connection and try again.")

def get_user_choice() -> int:
    """Get and validate user's choice of version.
    
    Returns:
        int: User's validated choice (1, 2, or 3)
    """
    while True:
        try:
            choice = int(input("\nEnter your choice (1, 2, or 3): "))
            if choice in [1, 2, 3]:
                return choice
            print("Please enter either 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number (1, 2, or 3).")

def initialize_camera():
    """Initialize and validate camera connection.
    
    Returns:
        cv2.VideoCapture: Initialized camera object or None if failed
    """
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Unable to access camera")
        return None
    return cap

if __name__ == "__main__":
    main() 