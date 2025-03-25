import os
import time
import cv2
import numpy as np
import dlib
import argparse
import platform


class EyeTrackingHeatmap:
    
    def __init__(self, stimulus_path=None, camera_id=1, resolution=None): #camera_id=1 change back to 0
        self.camera_id = camera_id
        self.cap = None
        
        # Get screen resolution if not provided
        if resolution is None:
            self.resolution = self._get_screen_resolution()
        else:
            self.resolution = resolution
            
        print(f"Using resolution: {self.resolution[0]}x{self.resolution[1]}")

        # Load Dlib face detector & shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.stimulus_path = stimulus_path
        self.stimulus = self.load_stimulus()
        
        self.gaze_points = []
        self.heatmap = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.float32)
        self.running = False
        
        # Eye visualization window
        self.eye_vis_enabled = True
        self.eye_vis_frame = None
        
        # Calibration Data
        self.calibrated = False
        
        # Generate a 3x3 grid of calibration points (9 points)
        self.generate_calibration_points()
        
        self.left_eye_corners = []
        self.right_eye_corners = []

    def _get_screen_resolution(self):
        system = platform.system()
        try:
            if system == "Windows":
                import ctypes
                user32 = ctypes.windll.user32
                user32.SetProcessDPIAware()
                width = user32.GetSystemMetrics(0)
                height = user32.GetSystemMetrics(1)
                return (width, height)

            elif system == "Darwin":
                import Quartz
                main_display = Quartz.CGMainDisplayID()
                width = Quartz.CGDisplayPixelsWide(main_display)
                height = Quartz.CGDisplayPixelsHigh(main_display)
                print(f"Screen resolution: {width}x{height}")
                return (width, height)

            elif system == "Linux":
                import subprocess
                output = subprocess.check_output("xrandr | grep '*'", shell=True).decode()
                res = output.strip().split()[0]
                width, height = map(int, res.split('x'))
                return (width, height)

            else:
                return (1920, 1080)  # fallback

        except Exception as e:
            print(f"Error detecting resolution: {e}")
            return (1920, 1080)  # fallback

    def _create_fullscreen_window(self, title):
        """Create a fullscreen window with a title"""
        # Create fullscreen window 
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return self.resolution

    def load_stimulus(self):
        """Load stimulus image and resize to match full screen resolution"""
        if self.stimulus_path and os.path.exists(self.stimulus_path):
            # Load stimulus image
            stimulus = cv2.imread(self.stimulus_path)
            
            # Get aspect ratio of the image and the screen
            img_height, img_width = stimulus.shape[:2]
            img_aspect = img_width / img_height
            screen_aspect = self.resolution[0] / self.resolution[1]
            
            # Resize image to fill screen while maintaining aspect ratio
            if img_aspect > screen_aspect:  # Image is wider than screen
                new_height = self.resolution[1]
                new_width = int(new_height * img_aspect)
            else:  # Image is taller than screen
                new_width = self.resolution[0]
                new_height = int(new_width / img_aspect)
                
            # Resize the image
            resized = cv2.resize(stimulus, (new_width, new_height))
            
            # Create a blank canvas of the screen size
            canvas = np.ones((self.resolution[1], self.resolution[0], 3), dtype=np.uint8) * 0
            
            # Calculate position to center the resized image on canvas
            y_offset = max(0, (self.resolution[1] - new_height) // 2)
            x_offset = max(0, (self.resolution[0] - new_width) // 2)
            
            # Place the resized image on the canvas
            # Handle case where resized image is larger than canvas
            y_end = min(y_offset + new_height, self.resolution[1])
            x_end = min(x_offset + new_width, self.resolution[0])
            canvas[y_offset:y_end, x_offset:x_end] = resized[:y_end-y_offset, :x_end-x_offset]
            
            return canvas
        
        # If no stimulus, create white canvas of exact screen size
        return np.ones((self.resolution[1], self.resolution[0], 3), dtype=np.uint8) * 255

    def start_camera(self):
        """Initialize the camera with error handling"""
        try:
            print(f"Attempting to open camera {self.camera_id}...")
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {self.camera_id}")
            
            # Test camera by reading a frame
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Could not read from camera")
            
            print(f"\nCamera initialized successfully at {frame.shape[1]}x{frame.shape[0]}")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            if self.cap is not None:
                self.cap.release()
            return False


    # Helper function to process each eye
    def process_eye(self, frame, eye_pts, is_left):
        """Process each eye to get the bounding box and eye region"""
        # Get bounding box
        min_x = int(np.min(eye_pts[:, 0]))
        max_x = int(np.max(eye_pts[:, 0]))
        
        # Use inner points for vertical bounds (more precise)
        top_points = eye_pts[1:3]  # landmarks 37-38 or 43-44
        bottom_points = eye_pts[4:6]  # landmarks 40-41 or 46-47
        
        min_y = int(min(p[1] for p in top_points))
        max_y = int(max(p[1] for p in bottom_points))
        

        # Add horizontal and vertical padding
        eye_width = max_x - min_x
        eye_height = max_y - min_y
        
        # Horizontal padding 
        h_padding = int(eye_width * 0.10)
        # Vertical padding 
        v_padding = int(eye_height * 0.10)
        
        # Apply padding with boundary checks
        min_x = max(0, min_x - h_padding)
        max_x = min(frame.shape[1], max_x + h_padding)
        min_y = max(0, min_y - v_padding)
        max_y = min(frame.shape[0], max_y + v_padding)
    
        
        # Draw eye outline and landmarks
        cv2.polylines(frame, [eye_pts], True, (0, 0, 255), 2)
        for pt in eye_pts:
            cv2.circle(frame, (pt[0], pt[1]), 2, (0, 0, 255), -1)
        
        # Draw bounding box
        label = "Left Eye" if is_left else "Right Eye"
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        cv2.putText(frame, label, (min_x, min_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Crop eye region
        eye_region = frame[min_y:max_y, min_x:max_x].copy()
        
        return eye_region, (min_x, min_y, max_x, max_y)

    def detect_eyes(self, frame):
        """
        Detects and processes eyes using Dlib's shape predictor.
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (left_eye_region, right_eye_region, annotated_frame)
                   Returns (None, None, frame) if detection fails
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            
            if len(faces) == 0:
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return None, None, frame
            
            # Process only the largest face if multiple faces detected
            if len(faces) > 1:
                faces = [max(faces, key=lambda rect: rect.area())]
                
            face = faces[0]
            landmarks = self.predictor(gray, face)

            # Extract eye landmarks
            left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

            # Process both eyes
            left_eye_region, left_bounds = self.process_eye(frame, left_eye_pts, True)
            right_eye_region, right_bounds = self.process_eye(frame, right_eye_pts, False)

            # Draw face detection info
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add pupil detection if eye regions are valid
            if left_eye_region.size > 0 and right_eye_region.size > 0:
                for eye_region, bounds, is_left in [
                    (left_eye_region, left_bounds, True),
                    (right_eye_region, right_bounds, False)
                ]:
                    pupil = self.get_pupil_center(eye_region)
                    if pupil:
                        pupil_x = bounds[0] + pupil[0]
                        pupil_y = bounds[1] + pupil[1]
                        cv2.circle(frame, (pupil_x, pupil_y), 3, (0, 255, 0), -1)
                        label = "Left" if is_left else "Right"
                        cv2.putText(frame, f"{label} Pupil", (pupil_x + 5, pupil_y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            return left_eye_region, right_eye_region, frame

        except Exception as e:
            print(f"Warning: Error in eye detection: {str(e)}")
            cv2.putText(frame, "Detection error", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None, None, frame

    def get_pupil_center(self, roi):
        """
        Detect the center of the pupil in the given eye region.
        
        Args:
            roi: Region of interest (eye region image)
            
        Returns:
            tuple: (x, y) coordinates of pupil center. Returns center of frame if detection fails.
        """
        try:
            height, width = roi.shape[:2]
            
            # Convert to grayscale first (if not already)
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # Increased kernel size for better noise reduction

            # Use adaptive thresholding with optimized parameters
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,  # Changed to MEAN for better pupil detection
                cv2.THRESH_BINARY_INV,
                11, 3  # Slightly adjusted parameters
            )

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return (width // 2, height // 2)

            # Filter contours by area to avoid small noise
            min_area = (width * height) * 0.01  # 1% of eye region
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

            if not valid_contours:
                return (width // 2, height // 2)

            # Get the darkest contour (likely to be the pupil)
            c = max(valid_contours, key=cv2.contourArea)
            
            # Calculate center using moments
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)

        except Exception as e:
            print(f"Warning: Error in pupil detection: {str(e)}")

        # Fallback to center if anything fails
        return (width // 2, height // 2)

    def get_relative_pupil_position(self, pupil_center, eye_region):
        """
        Calculate the relative position of the pupil within the eye.
        Returns normalized coordinates (-1 to 1) where (0,0) is the center of the eye.
        """
        height, width = eye_region.shape[:2]
        
        # Calculate the center of the eye region
        eye_center_x = width // 2
        eye_center_y = height // 2
        
        # Calculate the relative position of the pupil (-1 to 1)
        # where (0,0) is the center of the eye
        rel_x = (pupil_center[0] - eye_center_x) / (width / 2)
        rel_y = (pupil_center[1] - eye_center_y) / (height / 2)
        
        # Clamp values to [-1, 1] range
        rel_x = max(-1.0, min(1.0, rel_x))
        rel_y = max(-1.0, min(1.0, rel_y))
        
        return (rel_x, rel_y)
         
    def calibrate(self):
        """Calibrate eye tracking by mapping eye positions to screen corners"""
        self.eye_corners = []
        
        # Create full screen window
        self._create_fullscreen_window("Calibration")

        # Show calibration instructions
        instructions = np.ones((self.resolution[1], self.resolution[0], 3), dtype=np.uint8) * 0
        cv2.putText(
            instructions, "Eye Tracking Calibration", 
            (int(self.resolution[0] / 2) - 200, int(self.resolution[1] / 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2
        )
        cv2.putText(
            instructions, "Please look at each red circle as it appears", 
            (int(self.resolution[0] / 2) - 300, int(self.resolution[1] / 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        cv2.putText(
            instructions, "Press any key to begin", 
            (int(self.resolution[0] / 2) - 200, int(self.resolution[1] / 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        
        cv2.imshow("Calibration", instructions)
        cv2.waitKey(0)

        # Initialize storage for relative pupil positions
        self.left_eye_relative_positions = []
        self.right_eye_relative_positions = []
        
        # Initialize eye visualization window
        self.create_eye_visualization_window()

        for i, corner in enumerate(self.screen_corners):
            calibration_display = self.stimulus.copy()
            
            # Add countdown before showing the target
            for countdown in range(3, 0, -1):
                countdown_display = self.stimulus.copy()
                # Position countdown text in the center of the screen
                text_x = int(self.resolution[0] / 2 - 200)  # Center horizontally
                text_y = int(self.resolution[1] / 2)        # Center vertically
                cv2.putText(
                    countdown_display, f"Look at the {self.calibration_point_labels[i]} point in {countdown}...",
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
                )
                cv2.imshow("Calibration", countdown_display)
                cv2.waitKey(1000)
            
            # Show the target
            cv2.circle(calibration_display, (int(corner[0]), int(corner[1])), 25, (0, 0, 255), -1)
            
            # Position instruction text in the center of the screen
            center_x = int(self.resolution[0] / 2 - 200)
            center_y = int(self.resolution[1] / 2)
            cv2.putText(
                calibration_display, f"Focus on the RED circle ({self.calibration_point_labels[i]})",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
            )

            cv2.imshow("Calibration", calibration_display)
            cv2.waitKey(500)

            # Detection phase
            left_eye_positions = []
            right_eye_positions = []
            left_rel_positions = []
            right_rel_positions = []
            start_time = time.time()
            detection_duration = 3  # seconds
            detection_frames = 0
            successful_detections = 0
            
            # Create feedback display
            feedback_display = calibration_display.copy()
            
            while time.time() - start_time < detection_duration:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                left_eye, right_eye, frame = self.detect_eyes(frame)

                # Update eye visualization window
                self.update_eye_visualization(frame, left_eye, right_eye)

                # Check if eyes were detected
                if left_eye is None or right_eye is None:
                    continue

                left_eye_pupil = self.get_pupil_center(left_eye)
                right_eye_pupil = self.get_pupil_center(right_eye)

                # Calculate relative pupil positions within eye
                left_rel_pupil = self.get_relative_pupil_position(left_eye_pupil, left_eye)
                right_rel_pupil = self.get_relative_pupil_position(right_eye_pupil, right_eye)

                detection_frames += 1
                
                # Store both absolute and relative positions
                left_eye_positions.append(left_eye_pupil)
                right_eye_positions.append(right_eye_pupil)
                left_rel_positions.append(left_rel_pupil)
                right_rel_positions.append(right_rel_pupil)
                
                # Show detection progress in a small window (optional visual feedback)
                cv2.putText(frame, f"Collecting eye data: {len(left_eye_positions)}/{detection_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            
                
                successful_detections += 1
                
            # Calculate median pupil positions from collected frames
            if successful_detections > 0:
                # Convert lists to numpy arrays for easier calculation
                left_eye_positions = np.array(left_eye_positions)
                right_eye_positions = np.array(right_eye_positions)
                left_rel_positions = np.array(left_rel_positions)
                right_rel_positions = np.array(right_rel_positions)
                
                # Calculate median relative positions (more robust than mean against outliers)
                median_left_rel = (
                    np.median(left_rel_positions[:, 0]), 
                    np.median(left_rel_positions[:, 1])
                )
                
                median_right_rel = (
                    np.median(right_rel_positions[:, 0]), 
                    np.median(right_rel_positions[:, 1])
                )
                
                print(f"Point {i+1}/{len(self.screen_corners)}: Collected {successful_detections} eye positions")
                print(f"Left eye relative median: {median_left_rel}")
                print(f"Right eye relative median: {median_right_rel}")
                
                # Store relative pupil positions for this corner                
                self.left_eye_relative_positions.append(median_left_rel)
                self.right_eye_relative_positions.append(median_right_rel)
                
                # Keep original absolute positions for backward compatibility
                self.left_eye_corners.append((
                    int(np.median(left_eye_positions[:, 0])), 
                    int(np.median(left_eye_positions[:, 1]))
                ))
                self.right_eye_corners.append((
                    int(np.median(right_eye_positions[:, 0])), 
                    int(np.median(right_eye_positions[:, 1]))
                ))
    
                
                # Show success message
                success_display = calibration_display.copy()
                cv2.putText(
                    success_display, f"Position {i+1}/{len(self.screen_corners)} calibrated successfully!",
                    (int(self.resolution[0] / 4), int(self.resolution[1] - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                
                # Add instruction to press any key to continue to the next calibration point
                cv2.putText(
                    success_display, "Press any key to continue to the next point",
                    (int(self.resolution[0] / 4), int(self.resolution[1] - 100)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                
                cv2.imshow("Calibration", success_display)
                # Wait for key press before continuing to the next calibration point
                cv2.waitKey(0)
            else:
                print(f"Failed to detect eyes for point {i+1}")
                # Show failure message and wait for key press
                failure_display = calibration_display.copy()
                cv2.putText(
                    failure_display, f"Failed to detect eyes for point {i+1}",
                    (int(self.resolution[0] / 4), int(self.resolution[1] - 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
                cv2.putText(
                    failure_display, "Press any key to try the next point",
                    (int(self.resolution[0] / 4), int(self.resolution[1] - 100)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
                cv2.imshow("Calibration", failure_display)
                cv2.waitKey(0)
                
        # After collecting all corners, check if calibration is complete
        if len(self.left_eye_relative_positions) == len(self.screen_corners) and len(self.right_eye_relative_positions) == len(self.screen_corners):
            self.calibrated = True
            
            # Show final success message
            final_display = self.stimulus.copy()
            cv2.putText(
                final_display, "Calibration completed successfully!",
                (int(self.resolution[0] / 4), int(self.resolution[1] / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
            )
            cv2.putText(
                final_display, "Press any key to continue",
                (int(self.resolution[0] / 4), int(self.resolution[1] / 2) + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.imshow("Calibration", final_display)
            cv2.waitKey(0)
            
            cv2.destroyWindow("Calibration")
            print("Calibration completed successfully!")
            return True
        
        # Show failure message if calibration is incomplete
        final_display = self.stimulus.copy()
        cv2.putText(
            final_display, "Calibration failed. Not all points were calibrated.",
            (int(self.resolution[0] / 4), int(self.resolution[1] / 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        cv2.putText(
            final_display, "Press any key to continue",
            (int(self.resolution[0] / 4), int(self.resolution[1] / 2) + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        cv2.imshow("Calibration", final_display)
        cv2.waitKey(0)
        cv2.destroyWindow("Calibration")
        
        print("Calibration failed. Please try again.")
        return False

    def create_eye_visualization_window(self):
        """Create a window to continuously display eye tracking visualization"""
        if self.eye_vis_enabled:
            # Create a named window that can be kept on top
            cv2.namedWindow("Eye Tracking Visualization", cv2.WINDOW_NORMAL)
            
            # Resize window to appropriate dimensions - wider to accommodate both views
            cv2.resizeWindow("Eye Tracking Visualization", 1280, 320)
            
            # Position the window at the top-right corner of the screen to avoid overlap with calibration
            cv2.moveWindow("Eye Tracking Visualization", max(0, self.resolution[0] - 1290), 10)
            
            # Start with blank frame
            self.eye_vis_frame = np.ones((320, 1280, 3), dtype=np.uint8) * 255
            cv2.imshow("Eye Tracking Visualization", self.eye_vis_frame)
            
            # Try to keep the window always on top (platform-specific)
            try:
                import platform
                system = platform.system()
                
                if system == "Windows":
                    import ctypes
                    hwnd = ctypes.windll.user32.FindWindowW(None, "Eye Tracking Visualization")
                    if hwnd != 0:
                        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
                elif system == "Darwin":  # macOS
                    import subprocess
                    cmd = """
                    osascript -e 'tell application "System Events" to set frontmost of every process whose name contains "python" to true'
                    """
                    subprocess.Popen(cmd, shell=True)
                    
            except Exception as e:
                print(f"Note: Could not set window to always on top: {e}")
                
            cv2.waitKey(1)
            
    def update_eye_visualization(self, frame, left_eye, right_eye):
        """Update the eye tracking visualization window with current eye data"""
        if not self.eye_vis_enabled:
            return
            
        # Create a blank white canvas - make it wider to accommodate both views
        vis_frame = np.ones((320, 1280, 3), dtype=np.uint8) * 255
        
        # Draw dividers
        cv2.line(vis_frame, (320, 0), (320, 320), (0, 0, 0), 2)  # Between left eye views
        cv2.line(vis_frame, (640, 0), (640, 320), (0, 0, 0), 2)  # Center divider
        cv2.line(vis_frame, (960, 0), (960, 320), (0, 0, 0), 2)  # Between right eye views
        
        # Display detected eyes with contours
        if left_eye is not None:
            # Resize left eye to fit in visualization window (max 300x150)
            h, w = left_eye.shape[:2]
            scale = min(300 / w, 150 / h)
            resized_left = cv2.resize(left_eye, None, fx=scale, fy=scale)
            
            # Position in left half
            h2, w2 = resized_left.shape[:2]
            x_offset_color = 10  # For color image
            x_offset_thresh = 330  # For threshold image
            y_offset = 10
            
            # Display color image with outlines
            vis_frame[y_offset:y_offset+h2, x_offset_color:x_offset_color+w2] = resized_left
            cv2.rectangle(vis_frame, (x_offset_color, y_offset), 
                         (x_offset_color+w2, y_offset+h2), (0, 0, 255), 2)
            
            # Create thresholded version
            gray_left = cv2.cvtColor(resized_left, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_left, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            # Display thresholded image
            vis_frame[y_offset:y_offset+h2, x_offset_thresh:x_offset_thresh+w2] = thresh_colored
            cv2.rectangle(vis_frame, (x_offset_thresh, y_offset), 
                         (x_offset_thresh+w2, y_offset+h2), (0, 0, 255), 2)
            
            # Find and draw pupil on both views
            pupil_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            left_pupil = self.get_pupil_center(left_eye)
            pupil_x = int(left_pupil[0] * scale)
            pupil_y = int(left_pupil[1] * scale)
            
            # Draw pupil crosshair on color image
            cv2.circle(vis_frame, (x_offset_color + pupil_x, y_offset + pupil_y), 5, (0, 255, 0), -1)
            cv2.line(vis_frame, (x_offset_color + pupil_x - 10, y_offset + pupil_y), 
                     (x_offset_color + pupil_x + 10, y_offset + pupil_y), (0, 255, 0), 2)
            cv2.line(vis_frame, (x_offset_color + pupil_x, y_offset + pupil_y - 10), 
                     (x_offset_color + pupil_x, y_offset + pupil_y + 10), (0, 255, 0), 2)
            
            # Draw pupil crosshair on threshold image
            cv2.circle(vis_frame, (x_offset_thresh + pupil_x, y_offset + pupil_y), 5, (0, 255, 0), -1)
            cv2.line(vis_frame, (x_offset_thresh + pupil_x - 10, y_offset + pupil_y), 
                     (x_offset_thresh + pupil_x + 10, y_offset + pupil_y), (0, 255, 0), 2)
            cv2.line(vis_frame, (x_offset_thresh + pupil_x, y_offset + pupil_y - 10), 
                     (x_offset_thresh + pupil_x, y_offset + pupil_y + 10), (0, 255, 0), 2)
            
            # Draw contours on threshold image
            cv2.drawContours(vis_frame[y_offset:y_offset+h2, x_offset_thresh:x_offset_thresh+w2], 
                           pupil_contours, -1, (0, 0, 255), 1)
            
            # Add labels
            cv2.putText(vis_frame, "Left Eye (Color)", (x_offset_color, y_offset + h2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(vis_frame, "Left Eye (Threshold)", (x_offset_thresh, y_offset + h2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Get and display relative position
            rel_pos = self.get_relative_pupil_position(left_pupil, left_eye)
            cv2.putText(vis_frame, f"Rel: ({rel_pos[0]:.2f}, {rel_pos[1]:.2f})", 
                       (x_offset_color, y_offset + h2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
        if right_eye is not None:
            # Resize right eye to fit in visualization window (max 300x150)
            h, w = right_eye.shape[:2]
            scale = min(300 / w, 150 / h)
            resized_right = cv2.resize(right_eye, None, fx=scale, fy=scale)
            
            # Position in right half
            h2, w2 = resized_right.shape[:2]
            x_offset_color = 650  # For color image
            x_offset_thresh = 970  # For threshold image
            y_offset = 10
            
            # Display color image with outlines
            vis_frame[y_offset:y_offset+h2, x_offset_color:x_offset_color+w2] = resized_right
            cv2.rectangle(vis_frame, (x_offset_color, y_offset), 
                         (x_offset_color+w2, y_offset+h2), (0, 0, 255), 2)
            
            # Create thresholded version
            gray_right = cv2.cvtColor(resized_right, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_right, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            # Display thresholded image
            vis_frame[y_offset:y_offset+h2, x_offset_thresh:x_offset_thresh+w2] = thresh_colored
            cv2.rectangle(vis_frame, (x_offset_thresh, y_offset), 
                         (x_offset_thresh+w2, y_offset+h2), (0, 0, 255), 2)
            
            # Find and draw pupil on both views
            pupil_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            right_pupil = self.get_pupil_center(right_eye)
            pupil_x = int(right_pupil[0] * scale)
            pupil_y = int(right_pupil[1] * scale)
            
            # Draw pupil crosshair on color image
            cv2.circle(vis_frame, (x_offset_color + pupil_x, y_offset + pupil_y), 5, (0, 255, 0), -1)
            cv2.line(vis_frame, (x_offset_color + pupil_x - 10, y_offset + pupil_y), 
                     (x_offset_color + pupil_x + 10, y_offset + pupil_y), (0, 255, 0), 2)
            cv2.line(vis_frame, (x_offset_color + pupil_x, y_offset + pupil_y - 10), 
                     (x_offset_color + pupil_x, y_offset + pupil_y + 10), (0, 255, 0), 2)
            
            # Draw pupil crosshair on threshold image
            cv2.circle(vis_frame, (x_offset_thresh + pupil_x, y_offset + pupil_y), 5, (0, 255, 0), -1)
            cv2.line(vis_frame, (x_offset_thresh + pupil_x - 10, y_offset + pupil_y), 
                     (x_offset_thresh + pupil_x + 10, y_offset + pupil_y), (0, 255, 0), 2)
            cv2.line(vis_frame, (x_offset_thresh + pupil_x, y_offset + pupil_y - 10), 
                     (x_offset_thresh + pupil_x, y_offset + pupil_y + 10), (0, 255, 0), 2)
            
            # Draw contours on threshold image
            cv2.drawContours(vis_frame[y_offset:y_offset+h2, x_offset_thresh:x_offset_thresh+w2], 
                           pupil_contours, -1, (0, 0, 255), 1)
            
            # Add labels
            cv2.putText(vis_frame, "Right Eye (Color)", (x_offset_color, y_offset + h2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(vis_frame, "Right Eye (Threshold)", (x_offset_thresh, y_offset + h2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Get and display relative position
            rel_pos = self.get_relative_pupil_position(right_pupil, right_eye)
            cv2.putText(vis_frame, f"Rel: ({rel_pos[0]:.2f}, {rel_pos[1]:.2f})", 
                       (x_offset_color, y_offset + h2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add status info at the bottom
        detection_status = "Eyes Detected" if (left_eye is not None and right_eye is not None) else "Eyes Not Detected"
        status_color = (0, 255, 0) if (left_eye is not None and right_eye is not None) else (0, 0, 255)
        
        cv2.putText(vis_frame, f"Status: {detection_status}", 
                   (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Display timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(vis_frame, f"Time: {timestamp}", 
                   (1000, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Update the visualization window
        self.eye_vis_frame = vis_frame
        cv2.imshow("Eye Tracking Visualization", vis_frame)
        
        # Attempt to bring window to front periodically
        current_sec = int(timestamp.split(':')[2])
        if current_sec % 5 == 0:  # Every 5 seconds
            try:
                import platform
                system = platform.system()
                
                if system == "Windows":
                    import ctypes
                    hwnd = ctypes.windll.user32.FindWindowW(None, "Eye Tracking Visualization")
                    if hwnd != 0:
                        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
                elif system == "Darwin":  # macOS
                    import subprocess
                    cmd = """
                    osascript -e 'tell application "System Events" to set frontmost of every process whose name contains "python" to true'
                    """
                    subprocess.Popen(cmd, shell=True)
            except Exception:
                pass  # Silently ignore errors in window management
        
        cv2.waitKey(1)

    def map_eye_to_screen(self, pupil_center, eye_region, is_left_eye=True):
        """Maps detected eye position to screen coordinates using homography based on relative pupil position"""
        
        try:
            if not self.calibrated:
                return None
            
            # Calculate relative position of pupil within eye
            rel_pupil = self.get_relative_pupil_position(pupil_center, eye_region)
            
            # Choose which calibration data to use based on which eye we're mapping
            rel_eye_corners = self.left_eye_relative_positions if is_left_eye else self.right_eye_relative_positions
            
            # Calculate homography based on the relative pupil positions
            h, _ = cv2.findHomography(np.array(rel_eye_corners), np.array(self.screen_corners))
            
            # Transform the relative pupil position to screen coordinates
            rel_pupil_array = np.array([rel_pupil], dtype=np.float32).reshape(-1, 1, 2)
            screen_pos = cv2.perspectiveTransform(rel_pupil_array, h)
            
            return (int(screen_pos[0][0][0]), int(screen_pos[0][0][1]))
        except Exception as e:
            print(f"Warning: Error in eye-to-screen mapping: {e}")
            return None

    def verify_calibration(self):
        """Verify calibration by asking user to look at random points and checking accuracy"""
        if not self.calibrated:
            print("Calibration must be completed before verification.")
            return False
            
        # Ensure eye visualization window is created
        self.create_eye_visualization_window()
            
        # Create fullscreen window
        self._create_fullscreen_window("Verification")
        
        # Show instructions
        instructions = np.ones((self.resolution[1], self.resolution[0], 3), dtype=np.uint8) * 255
        cv2.putText(
            instructions, "Calibration Verification", 
            (int(self.resolution[0] / 4), int(self.resolution[1] / 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2
        )
        cv2.putText(
            instructions, "Please look at each target as it appears", 
            (int(self.resolution[0] / 6), int(self.resolution[1] / 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        cv2.putText(
            instructions, "Press any key to begin", 
            (int(self.resolution[0] / 3), int(2 * self.resolution[1] / 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
        )
        cv2.imshow("Verification", instructions)
        cv2.waitKey(0)
        
        # Generate verification points - first point at center of screen, then 4 random points
        verification_points = []
        
        # First point: center of screen
        center_point = (int(self.resolution[0]/2), int(self.resolution[1]/2))
        verification_points.append(center_point)
        
        # Remaining 4 random points
        np.random.seed(42)  # For reproducibility
        for _ in range(4):
            x = np.random.randint(50, self.resolution[0] - 50)
            y = np.random.randint(50, self.resolution[1] - 50)
            verification_points.append((x, y))
        
        errors = []
        
        for i, point in enumerate(verification_points):
            verification_display = self.stimulus.copy()
            
            # Add countdown before showing the target
            for countdown in range(3, 0, -1):
                countdown_display = self.stimulus.copy()
                cv2.putText(
                    countdown_display, f"Look at the next target in {countdown}...",
                    (int(self.resolution[0] / 4), int(self.resolution[1] / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
                )
                cv2.imshow("Verification", countdown_display)
                cv2.waitKey(1000)
            
            # Show the target
            cv2.circle(verification_display, point, 25, (0, 0, 255), -1)
            cv2.putText(
                verification_display, f"Focus on the RED circle (Point {i+1}/5)",
                (int(self.resolution[0] / 4), 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )
            cv2.imshow("Verification", verification_display)
            
            # Collect eye positions
            eye_positions = []
            start_time = time.time()
            duration = 3  # seconds
            
            while time.time() - start_time < duration:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                left_eye, right_eye, frame = self.detect_eyes(frame)
                
                # Update eye visualization
                self.update_eye_visualization(frame, left_eye, right_eye)
                
                # Create a new copy of verification display for each frame
                feedback = verification_display.copy()
                
                # Check if eyes were detected successfully
                if left_eye is not None and right_eye is not None:
                    # Get pupil centers for both eyes
                    left_pupil = self.get_pupil_center(left_eye)
                    right_pupil = self.get_pupil_center(right_eye)
                    
                    # Map each eye position to screen coordinates using relative pupil positions
                    left_screen_pos = self.map_eye_to_screen(left_pupil, left_eye, is_left_eye=True)
                    right_screen_pos = self.map_eye_to_screen(right_pupil, right_eye, is_left_eye=False)
                    
                    # Use average of both eyes if both were mapped successfully
                    if left_screen_pos and right_screen_pos:
                        # Calculate current gaze position
                        current_gaze = (
                            (left_screen_pos[0] + right_screen_pos[0]) // 2,
                            (left_screen_pos[1] + right_screen_pos[1]) // 2
                        )
                        
                        # Store for metrics calculation
                        eye_positions.append(current_gaze)
                        
                        # Draw the current gaze position with a more visible cursor
                        # Crosshair cursor
                        cv2.circle(feedback, current_gaze, 10, (0, 255, 255), 2)  # Yellow circle
                        cv2.line(feedback, (current_gaze[0] - 15, current_gaze[1]), 
                                (current_gaze[0] + 15, current_gaze[1]), (0, 255, 255), 2)
                        cv2.line(feedback, (current_gaze[0], current_gaze[1] - 15), 
                                (current_gaze[0], current_gaze[1] + 15), (0, 255, 255), 2)
                        
                        # Display coordinates
                        cv2.putText(
                            feedback, f"({current_gaze[0]}, {current_gaze[1]})",
                            (current_gaze[0] + 15, current_gaze[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                        )
                        
                        # Calculate and display real-time error
                        rt_error = np.sqrt((current_gaze[0] - point[0])**2 + (current_gaze[1] - point[1])**2)
                        cv2.putText(
                            feedback, f"Current Error: {rt_error:.1f} px",
                            (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
                        )
                        
                        # Visualize relative pupil positions for debugging
                        left_rel = self.get_relative_pupil_position(left_pupil, left_eye)
                        right_rel = self.get_relative_pupil_position(right_pupil, right_eye)
                        
                        cv2.putText(
                            feedback, f"L-Eye Rel: ({left_rel[0]:.2f}, {left_rel[1]:.2f})",
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                        )
                        cv2.putText(
                            feedback, f"R-Eye Rel: ({right_rel[0]:.2f}, {right_rel[1]:.2f})",
                            (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                        )
                
                # Show feedback
                cv2.putText(
                    feedback, f"Keep looking at the red target ({i+1}/5)",
                    (50, self.resolution[1] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                
                # Show progress bar
                progress = int((time.time() - start_time) / duration * 100)
                cv2.rectangle(feedback, (50, self.resolution[1] - 50), 
                             (50 + int(progress * 3), self.resolution[1] - 30), (0, 255, 0), -1)
                
                # Add detection status
                status = "DETECTED" if len(eye_positions) > 0 else "SEARCHING..."
                color = (0, 255, 0) if status == "DETECTED" else (0, 0, 255)
                cv2.putText(
                    feedback, f"Eye Status: {status} ({len(eye_positions)} points)",
                    (50, self.resolution[1] - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )
                
                # Small inset view of eye tracking
                if frame is not None:
                    # Scale down frame to a small inset
                    inset_height = 150
                    inset_width = int(frame.shape[1] * inset_height / frame.shape[0])
                    inset = cv2.resize(frame, (inset_width, inset_height))
                    
                    # Position in top-right corner
                    x_offset = self.resolution[0] - inset_width - 20
                    y_offset = 20
                    
                    # Create a small border around the inset
                    cv2.rectangle(feedback, (x_offset-2, y_offset-2), 
                                 (x_offset+inset_width+2, y_offset+inset_height+2), (0, 0, 0), 2)
                    
                    # Place the inset on the feedback display
                    feedback[y_offset:y_offset+inset_height, x_offset:x_offset+inset_width] = inset
                
                cv2.imshow("Verification", feedback)
                cv2.waitKey(1)
            
            # Calculate error
            if len(eye_positions) > 0:
                eye_positions = np.array(eye_positions)
                median_pos = (int(np.median(eye_positions[:, 0])), int(np.median(eye_positions[:, 1])))
                
                # Calculate Euclidean distance (error)
                error = np.sqrt((median_pos[0] - point[0])**2 + (median_pos[1] - point[1])**2)
                errors.append(error)
                
                # Show error
                result_display = self.stimulus.copy()
                cv2.circle(result_display, point, 25, (0, 0, 255), -1)  # Target
                cv2.circle(result_display, median_pos, 15, (255, 0, 0), -1)  # Detected gaze
                cv2.line(result_display, point, median_pos, (0, 255, 0), 2)  # Error line
                
                cv2.putText(
                    result_display, f"Point {i+1}/5 - Error: {error:.1f} pixels",
                    (int(self.resolution[0] / 4), int(self.resolution[1] / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
                )
                cv2.imshow("Verification", result_display)
                cv2.waitKey(2000)
                
                # Special message for first point (center point)
                if i == 0 and error > 100:
                    warning_display = self.stimulus.copy()
                    cv2.putText(
                        warning_display, "Center point error is high!",
                        (int(self.resolution[0] / 4), int(self.resolution[1] / 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )
                    cv2.putText(
                        warning_display, "You may need to recalibrate",
                        (int(self.resolution[0] / 4), int(self.resolution[1] / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )
                    cv2.imshow("Verification", warning_display)
                    cv2.waitKey(2000)
        
        # Calculate overall accuracy
        if len(errors) > 0:
            mean_error = np.mean(errors)
            
            # Show final results
            final_display = self.stimulus.copy()
            status = "GOOD" if mean_error < 150 else "POOR"  # Increased threshold
            color = (0, 255, 0) if status == "GOOD" else (0, 0, 255)
            
            cv2.putText(
                final_display, f"Calibration Verification Complete",
                (int(self.resolution[0] / 4), int(self.resolution[1] / 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )
            cv2.putText(
                final_display, f"Average Error: {mean_error:.1f} pixels",
                (int(self.resolution[0] / 4), int(self.resolution[1] / 2) - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )
            cv2.putText(
                final_display, f"Calibration Status: {status}",
                (int(self.resolution[0] / 4), int(self.resolution[1] / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
            )
            
            if status == "POOR":
                cv2.putText(
                    final_display, "Consider recalibrating for better results",
                    (int(self.resolution[0] / 4), int(self.resolution[1] / 2) + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )
                
            cv2.putText(
                final_display, "Press any key to continue",
                (int(self.resolution[0] / 4), int(2 * self.resolution[1] / 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )
            
            cv2.imshow("Verification", final_display)
            cv2.waitKey(0)
            cv2.destroyWindow("Verification")
            
            print(f"Verification complete. Average error: {mean_error:.1f} pixels")
            return mean_error < 150  # Increased threshold
            
        print("Verification failed - could not detect eyes consistently.")
        return False

    def track_eyes(self, duration=10):
        """Tracks eye movement and updates heatmap"""
        self.start_camera()

        if not self.calibrated:
            if not self.calibrate():
                return False
                
        # Add verification step after calibration
        print("Verifying calibration accuracy...")
        if not self.verify_calibration():
            print("Calibration verification failed or showed poor accuracy.")
            recalibrate = input("Would you like to recalibrate? (y/n): ")
            if recalibrate.lower() == 'y':
                if not self.calibrate() or not self.verify_calibration():
                    print("Recalibration failed. Exiting tracking.")
                    return False

        self.gaze_points = []
        self.heatmap = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.float32)
        
        # Create fullscreen window properly positioned
        self._create_fullscreen_window("Stimulus")
        cv2.imshow("Stimulus", self.stimulus)
        
        # Ensure eye visualization window is created
        self.create_eye_visualization_window()

        start_time = time.time()
        print(f"Starting eye tracking for {duration} seconds...")
        
        # Show countdown before tracking starts
        for countdown in range(3, 0, -1):
            countdown_display = self.stimulus.copy()
            cv2.putText(
                countdown_display, f"Eye tracking will begin in {countdown}...",
                (int(self.resolution[0] / 4), int(self.resolution[1] / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )
            cv2.imshow("Stimulus", countdown_display)
            cv2.waitKey(1000)
            
        # Show actual stimulus
        cv2.imshow("Stimulus", self.stimulus)
        
        # Variables for tracking
        last_positions = []  # Store last N positions for smoothing
        max_positions = 10   # Max number of positions to keep for smoothing
        tracking_status = "TRACKING"  # Status indicator
        detection_count = 0   # Counter for successful detections
        total_frames = 0      # Counter for total frames

        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            left_eye, right_eye, frame = self.detect_eyes(frame)
            
            # Update eye visualization
            self.update_eye_visualization(frame, left_eye, right_eye)
            
            total_frames += 1

            # Create a copy of stimulus for visualization
            stimulus_display = self.stimulus.copy()

            if left_eye is not None and right_eye is not None:
                # Get pupil centers for both eyes
                left_pupil = self.get_pupil_center(left_eye)
                right_pupil = self.get_pupil_center(right_eye)
                
                # Map each eye position to screen coordinates using relative position
                left_screen_pos = self.map_eye_to_screen(left_pupil, left_eye, is_left_eye=True)
                right_screen_pos = self.map_eye_to_screen(right_pupil, right_eye, is_left_eye=False)
                
                # Only proceed if both eye positions mapped successfully
                if left_screen_pos and right_screen_pos:
                    # Calculate average gaze position from both eyes
                    screen_pos = (
                        (left_screen_pos[0] + right_screen_pos[0]) // 2,
                        (left_screen_pos[1] + right_screen_pos[1]) // 2
                    )
                    
                    # Check if position is within screen boundaries
                    if (0 <= screen_pos[0] < self.resolution[0] and 
                        0 <= screen_pos[1] < self.resolution[1]):
                        detection_count += 1
                        tracking_status = "TRACKING"
                        
                        # Add to gaze points and update heatmap
                        self.gaze_points.append(screen_pos)
                        self._add_gaussian_to_heatmap(screen_pos)
                        
                        # Maintain a list of recent positions for smoothing
                        last_positions.append(screen_pos)
                        if len(last_positions) > max_positions:
                            last_positions.pop(0)
                        
                        # Show smooth gaze trail
                        if len(last_positions) > 1:
                            # Draw a trail of past positions with fading opacity
                            for i in range(len(last_positions) - 1):
                                alpha = (i + 1) / len(last_positions)  # Opacity increases with recency
                                cv2.line(stimulus_display, 
                                        last_positions[i], 
                                        last_positions[i+1], 
                                        (0, int(255 * alpha), int(255 * (1-alpha))), 
                                        thickness=2)
                        
                        # Draw current gaze position
                        cv2.circle(stimulus_display, screen_pos, 10, (0, 0, 255), -1)
                        
                        # Optionally, show individual eye positions with different colors
                        cv2.circle(stimulus_display, left_screen_pos, 5, (255, 0, 0), -1)  # Left eye in red
                        cv2.circle(stimulus_display, right_screen_pos, 5, (0, 255, 0), -1)  # Right eye in green
                    else:
                        tracking_status = "OUT OF BOUNDS"
                else:
                    tracking_status = "MAPPING FAILED"
            else:
                tracking_status = "EYES NOT DETECTED"
            
            # Add status information to the display
            # Calculate tracking rate
            tracking_rate = (detection_count / total_frames) * 100 if total_frames > 0 else 0
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)
            
            # Status color based on tracking rate
            if tracking_rate > 80:
                status_color = (0, 255, 0)  # Green for good tracking
            elif tracking_rate > 50:
                status_color = (0, 255, 255)  # Yellow for moderate tracking
            else:
                status_color = (0, 0, 255)  # Red for poor tracking
            
            # Add status overlay
            cv2.rectangle(stimulus_display, (10, 10), (400, 120), (255, 255, 255), -1)
            cv2.rectangle(stimulus_display, (10, 10), (400, 120), (0, 0, 0), 2)
            
            cv2.putText(stimulus_display, f"Status: {tracking_status}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(stimulus_display, f"Tracking Rate: {tracking_rate:.1f}%", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(stimulus_display, f"Time: {int(elapsed)}s / {duration}s", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add progress bar
            progress_width = int((elapsed / duration) * 350)
            cv2.rectangle(stimulus_display, (10, self.resolution[1] - 30), 
                         (10 + progress_width, self.resolution[1] - 10), (0, 255, 0), -1)
            
            # Show the updated stimulus
            cv2.imshow("Stimulus", stimulus_display)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Show completion message
        completion_display = self.stimulus.copy()
        cv2.putText(
            completion_display, "Eye tracking complete!",
            (int(self.resolution[0] / 4), int(self.resolution[1] / 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2
        )
        cv2.putText(
            completion_display, f"Collected {len(self.gaze_points)} gaze points",
            (int(self.resolution[0] / 4), int(self.resolution[1] / 2) + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            completion_display, "Press any key to continue to heatmap",
            (int(self.resolution[0] / 4), int(self.resolution[1] / 2) + 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Stimulus", completion_display)
        cv2.waitKey(0)

        print(f"Eye tracking complete! Collected {len(self.gaze_points)} gaze points.")
        self.cap.release()
        cv2.destroyAllWindows()
        return True

    def _add_gaussian_to_heatmap(self, center, sigma=50):
        """Adds a Gaussian blob to heatmap at given center"""
        y, x = np.meshgrid(np.arange(self.resolution[1]), np.arange(self.resolution[0]), indexing="ij")
        dst = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        gaussian = np.exp(-(dst**2) / (2 * sigma**2))
        self.heatmap += gaussian

    def generate_heatmap(self, save_path="heatmap.png", alpha=0.6):
        """Generates and displays gaze heatmap with metrics"""
        if len(self.gaze_points) == 0:
            print("No gaze data collected.")
            return
        
        # Normalize heatmap for visualization
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # Create blended image with adjustable transparency
        blended = cv2.addWeighted(self.stimulus, 1.0 - alpha, heatmap_color, alpha, 0)
        
        # Calculate metrics
        gaze_points_array = np.array(self.gaze_points)
        
        # Find hotspots (areas with highest concentration)
        _, max_val, _, max_loc = cv2.minMaxLoc(self.heatmap)
        
        # Calculate distribution metrics
        x_coords = gaze_points_array[:, 0]
        y_coords = gaze_points_array[:, 1]
        mean_x, mean_y = np.mean(x_coords), np.mean(y_coords)
        std_x, std_y = np.std(x_coords), np.std(y_coords)
        
        # Find the boundaries of the gaze area (bounding box)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        # Draw metrics on the image
        # Create a semi-transparent overlay for metrics
        metrics_overlay = blended.copy()
        cv2.rectangle(metrics_overlay, (10, 10), (400, 200), (255, 255, 255), -1)
        blended = cv2.addWeighted(blended, 0.7, metrics_overlay, 0.3, 0, blended)
        
        # Add metrics text
        cv2.putText(blended, f"Total Gaze Points: {len(self.gaze_points)}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(blended, f"Hotspot: ({max_loc[0]}, {max_loc[1]})", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(blended, f"Mean Position: ({int(mean_x)}, {int(mean_y)})", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(blended, f"Gaze Dispersion: {int(std_x + std_y)} px", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(blended, f"Coverage: {int((max_x-min_x)*(max_y-min_y)/(self.resolution[0]*self.resolution[1])*100)}%", 
                   (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Mark the hotspot
        cv2.circle(blended, max_loc, 15, (0, 0, 255), 2)
        
        # Mark the mean gaze position
        cv2.circle(blended, (int(mean_x), int(mean_y)), 10, (255, 0, 0), 2)
        
        # Draw bounding box of gaze area
        cv2.rectangle(blended, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
        
        # Show interactive controls
        controls_text = "Press 'S' to save, '+'/'-' to adjust opacity, 'ESC' to exit"
        cv2.putText(blended, controls_text, 
                   (20, self.resolution[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Create fullscreen window properly positioned
        self._create_fullscreen_window("Gaze Heatmap")
        
        # Interactive display loop
        current_alpha = alpha
        while True:
            cv2.imshow("Gaze Heatmap", blended)
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('s') or key == ord('S'):  # Save
                cv2.imwrite(save_path, blended)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                metrics_path = f"heatmap_metrics_{timestamp}.txt"
                
                # Save detailed metrics to file
                with open(metrics_path, 'w') as f:
                    f.write(f"Eye Tracking Heatmap Metrics\n")
                    f.write(f"=========================\n")
                    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"Total Gaze Points: {len(self.gaze_points)}\n")
                    f.write(f"Hotspot Location: ({max_loc[0]}, {max_loc[1]})\n")
                    f.write(f"Mean Gaze Position: ({mean_x:.2f}, {mean_y:.2f})\n")
                    f.write(f"Standard Deviation X: {std_x:.2f} pixels\n")
                    f.write(f"Standard Deviation Y: {std_y:.2f} pixels\n")
                    f.write(f"Gaze Dispersion: {std_x + std_y:.2f} pixels\n")
                    f.write(f"Bounding Box: ({min_x:.0f}, {min_y:.0f}) to ({max_x:.0f}, {max_y:.0f})\n")
                    f.write(f"Area Coverage: {(max_x-min_x)*(max_y-min_y)/(self.resolution[0]*self.resolution[1])*100:.2f}%\n")
                
                print(f"Heatmap saved to {save_path}")
                print(f"Metrics saved to {metrics_path}")
                
            elif key == ord('+') or key == ord('='):  # Increase opacity
                current_alpha = min(1.0, current_alpha + 0.1)
                blended = cv2.addWeighted(self.stimulus, 1.0 - current_alpha, heatmap_color, current_alpha, 0)
                
                # Reapply metrics overlay
                metrics_overlay = blended.copy()
                cv2.rectangle(metrics_overlay, (10, 10), (400, 200), (255, 255, 255), -1)
                blended = cv2.addWeighted(blended, 0.7, metrics_overlay, 0.3, 0, blended)
                
                # Redraw all metrics text and markers
                cv2.putText(blended, f"Total Gaze Points: {len(self.gaze_points)}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(blended, f"Hotspot: ({max_loc[0]}, {max_loc[1]})", 
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(blended, f"Mean Position: ({int(mean_x)}, {int(mean_y)})", 
                           (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(blended, f"Gaze Dispersion: {int(std_x + std_y)} px", 
                           (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(blended, f"Coverage: {int((max_x-min_x)*(max_y-min_y)/(self.resolution[0]*self.resolution[1])*100)}%", 
                           (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.circle(blended, max_loc, 15, (0, 0, 255), 2)
                cv2.circle(blended, (int(mean_x), int(mean_y)), 10, (255, 0, 0), 2)
                cv2.rectangle(blended, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
                cv2.putText(blended, controls_text, 
                           (20, self.resolution[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
            elif key == ord('-') or key == ord('_'):  # Decrease opacity
                current_alpha = max(0.1, current_alpha - 0.1)
                blended = cv2.addWeighted(self.stimulus, 1.0 - current_alpha, heatmap_color, current_alpha, 0)
                
                # Reapply metrics overlay
                metrics_overlay = blended.copy()
                cv2.rectangle(metrics_overlay, (10, 10), (400, 200), (255, 255, 255), -1)
                blended = cv2.addWeighted(blended, 0.7, metrics_overlay, 0.3, 0, blended)
                
                # Redraw all metrics text and markers
                cv2.putText(blended, f"Total Gaze Points: {len(self.gaze_points)}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(blended, f"Hotspot: ({max_loc[0]}, {max_loc[1]})", 
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(blended, f"Mean Position: ({int(mean_x)}, {int(mean_y)})", 
                           (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(blended, f"Gaze Dispersion: {int(std_x + std_y)} px", 
                           (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(blended, f"Coverage: {int((max_x-min_x)*(max_y-min_y)/(self.resolution[0]*self.resolution[1])*100)}%", 
                           (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.circle(blended, max_loc, 15, (0, 0, 255), 2)
                cv2.circle(blended, (int(mean_x), int(mean_y)), 10, (255, 0, 0), 2)
                cv2.rectangle(blended, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 0), 2)
                cv2.putText(blended, controls_text, 
                           (20, self.resolution[1] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.destroyAllWindows()
        
        # Default save
        if not os.path.exists(save_path):
            cv2.imwrite(save_path, blended)
            print(f"Heatmap saved to {save_path}")
        
        return blended

    def generate_calibration_points(self):
        """Generate a grid of calibration points with horizontal and vertical lines through the center"""
        width, height = self.resolution
        
        # Center point
        center_x = width // 2
        center_y = height // 2
        
        
        # Create calibration points in a grid formation
        self.screen_corners = np.array([
            # Center point
            [center_x, center_y],
            
            # Horizontal line (left to right)
            [0, center_y], # Left-center
            [width, center_y], # Right-center
            
            # Vertical line (top to bottom)
            [center_x, 0], # Top-center
            [center_x, height], # Bottom-center
            
            # Corner points
            [0, 0], # Top-left
            [width, 0], # Top-right
            [0, height], # Bottom-left
            [width, height], # Bottom-right
        ])
        
        # Define text labels for each calibration point
        self.calibration_point_labels = [
            "center",
            "left-center", "right-center",
            "top-center", "bottom-center",
            "top-left", "top-right",
            "bottom-left", "bottom-right"
        ]


def main():
    """Main function with command line argument handling"""
    parser = argparse.ArgumentParser(description='Eye Tracking Heatmap Generator')
    parser.add_argument('--stimulus', type=str, help='Path to stimulus image file')
    parser.add_argument('--duration', type=int, default=10, help='Duration of eye tracking in seconds')
    parser.add_argument('--output', type=str, default='heatmap.png', help='Output file path for the heatmap')
    parser.add_argument('--resolution', type=str, help='Resolution of the heatmap (WxH). Defaults to screen resolution.')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID to use')
    parser.add_argument('--alpha', type=float, default=0.6, help='Opacity of heatmap overlay (0.1-1.0)')
    parser.add_argument('--skip-calibration-verification', action='store_true', help='Skip calibration verification')
    args = parser.parse_args()
    
    # Parse resolution if provided
    resolution = None
    if args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            resolution = (width, height)
        except ValueError:
            print(f"Invalid resolution format: {args.resolution}. Using screen resolution.")
            resolution = None
    
    # Create tracker instance
    try:
        tracker = EyeTrackingHeatmap(resolution=resolution)
    except Exception as e:
        print(f"Error creating tracker: {e}")
        return
    
    # Start camera first to make sure it's available
    if not tracker.start_camera():
        print("Failed to initialize camera. Please check your camera connection and try again.")
        return
    
    if args.stimulus:
        print(f"Stimulus: {args.stimulus}")
    print(f"Resolution: {tracker.resolution[0]}x{tracker.resolution[1]}")
    print(f"Duration: {args.duration} seconds")
    print(f"Output: {args.output}")
    print("\n")
    
    

    # Run calibration
    print("Starting calibration process...")
    if not tracker.calibrate():
        print("Calibration failed. Please try again with better lighting and positioning.")
        return
    
    # Verify calibration
    print("Verifying calibration accuracy...")
    if not tracker.verify_calibration() and not args.skip_calibration_verification:
        print("Calibration verification failed or showed poor accuracy.")
        return
    else:
        print("Warning: Proceeding with unverified calibration. Results may be inaccurate.")
    
    # Run tracking
    try:
        if tracker.track_eyes(duration=args.duration):
            print("Generating heatmap...")
            tracker.generate_heatmap(save_path=args.output, alpha=args.alpha)
            print("Eye tracking session completed successfully!")
        else:
            print("Eye tracking failed. Please check your camera and try again.")
    except Exception as e:
        print(f"Error during eye tracking: {e}")
    finally:
        # Clean up
        if tracker.cap is not None:
            tracker.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
