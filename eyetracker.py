import os
import time
import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt


class EyeTrackingHeatmap:
    def __init__(self, stimulus_path=None, camera_id=0, heatmap_resolution=(800, 600)):
        self.camera_id = camera_id
        self.cap = None
        
        # Load Dlib face detector & landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        self.heatmap_resolution = heatmap_resolution
        self.stimulus_path = stimulus_path
        self.stimulus = self.load_stimulus()
        
        self.gaze_points = []
        self.heatmap = np.zeros((heatmap_resolution[1], heatmap_resolution[0]), dtype=np.float32)
        self.running = False
        
    def load_stimulus(self):
        if self.stimulus_path and os.path.exists(self.stimulus_path):
            stimulus = cv2.imread(self.stimulus_path)
            return cv2.resize(stimulus, self.heatmap_resolution)
        return np.ones((self.heatmap_resolution[1], self.heatmap_resolution[0], 3), dtype=np.uint8) * 255
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception("Could not open camera.")
    
    def detect_eyes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        for face in faces:
            landmarks = self.predictor(gray, face)
            left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            
            # Draw a border around the eyes using a polygon shape
            cv2.polylines(frame, [left_eye_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(frame, [right_eye_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            left_eye = (landmarks.part(36).x, landmarks.part(36).y)
            right_eye = (landmarks.part(45).x, landmarks.part(45).y)
            eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            return eye_center
        return None
    
    def track_eyes(self, duration=10):
        self.start_camera()
        start_time = time.time()
        
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            eye_pos = self.detect_eyes(frame)
            
            if eye_pos:
                self.gaze_points.append(eye_pos)
                self._add_gaussian_to_heatmap(eye_pos)
                cv2.circle(frame, eye_pos, 10, (0, 255, 0), -1)
            
            cv2.imshow("Eye Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _add_gaussian_to_heatmap(self, center, sigma=50):
        y, x = np.meshgrid(
            np.arange(self.heatmap_resolution[1]),
            np.arange(self.heatmap_resolution[0]),
            indexing="ij",
        )
        dst = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        gaussian = np.exp(-(dst**2) / (2 * sigma**2))
        self.heatmap += gaussian
    
    def generate_heatmap(self, save_path="heatmap.png"):
        if len(self.gaze_points) == 0:
            print("No gaze data collected.")
            return
        
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_normalized = heatmap_normalized.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        blended = cv2.addWeighted(self.stimulus, 0.6, heatmap_color, 0.4, 0)
        cv2.imshow("Gaze Heatmap", blended)
        cv2.waitKey(0)
        cv2.imwrite(save_path, blended)
        print(f"Heatmap saved to {save_path}")
        

    

def main():
    stimulus_path = input("Enter stimulus image path (or leave blank for none): ").strip()
    tracker = EyeTrackingHeatmap(stimulus_path=stimulus_path)
    
    print("Tracking eyes for 10 seconds...")
    tracker.track_eyes(duration=10)
    
    print("Generating heatmap...")
    tracker.generate_heatmap()

    

if __name__ == "__main__":
    main()
