import cv2
from eyetracker import EyeTracker

tracker = EyeTracker()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tracker.refresh(frame)
    output = tracker.annotated_frame()

    if tracker.is_ready():
        cv2.putText(output, "Tracking eyes...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Eye Tracker (No dlib)", output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
