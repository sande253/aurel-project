import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert frame to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for pose detection
    results = pose.process(frame_rgb)

    # Draw pose landmarks and dress placeholder
    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get key landmarks (e.g., shoulders and hips for torso)
        h, w, _ = frame.shape
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate torso center
        torso_center_x = int((left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) * w / 4)
        torso_center_y = int((left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) * h / 4)

        # Draw a placeholder rectangle for the dress
        dress_width, dress_height = 100, 150  # Adjust based on dress model size
        top_left = (torso_center_x - dress_width // 2, torso_center_y - dress_height // 2)
        bottom_right = (torso_center_x + dress_width // 2, torso_center_y + dress_height // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        # TODO: Replace rectangle with 3D dress model rendering (see Step 4)

    # Display the frame
    cv2.imshow("AR Dress Try-On", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()