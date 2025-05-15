import cv2
import mediapipe as mp
import numpy as np
import trimesh

# Initialize Mediapipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Load 3D dress model (GLB) - Optional for now, since we're using a 2D texture
dress_model = None
def load_dress_model():
    global dress_model
    dress_model = trimesh.load("tshirt.glb")
    print("Loaded dress model:", dress_model)

# Load 2D dress image (PNG with transparency)
dress_image = cv2.imread("tshirt.png", cv2.IMREAD_UNCHANGED)  # Load with alpha channel
if dress_image is None:
    print("Error: Could not load tshirt.png. Please ensure the file exists.")
    exit()

# Resize dress image to fit the torso (adjust as needed)
dress_image = cv2.resize(dress_image, (100, 150))  # Match the placeholder size

# Function to render the 2D dress image on the frame
def render_dress(frame, dress_img, torso_x, torso_y):
    dress_h, dress_w = dress_img.shape[:2]
    x, y = int(torso_x - dress_w // 2), int(torso_y - dress_h // 2)

    # Ensure the dress stays within the frame boundaries
    y1, y2 = max(0, y), min(frame.shape[0], y + dress_h)
    x1, x2 = max(0, x), min(frame.shape[1], x + dress_w)

    # Calculate the corresponding region in the dress image
    dress_y1 = max(0, -y)
    dress_x1 = max(0, -x)
    dress_y2 = dress_y1 + (y2 - y1)
    dress_x2 = dress_x1 + (x2 - x1)

    # Extract the alpha channel (transparency)
    if dress_img.shape[2] == 4:  # Check if image has alpha channel
        alpha = dress_img[dress_y1:dress_y2, dress_x1:dress_x2, 3] / 255.0
        dress_rgb = dress_img[dress_y1:dress_y2, dress_x1:dress_x2, :3]
    else:
        alpha = np.ones((dress_y2 - dress_y1, dress_x2 - dress_x1), dtype=np.float32)
        dress_rgb = dress_img[dress_y1:dress_y2, dress_x1:dress_x2]

    # Overlay the dress on the frame using alpha blending
    for c in range(3):
        frame[y1:y2, x1:x2, c] = frame[y1:y2, x1:x2, c] * (1 - alpha) + dress_rgb[:, :, c] * alpha

# Start webcam
cap = cv2.VideoCapture(0)

# Load the dress model (optional for now)
load_dress_model()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert frame to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame for pose detection
    results = pose.process(frame_rgb)

    # Draw pose landmarks and dress
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get key landmarks (e.g., shoulders for torso)
        h, w, _ = frame.shape
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        torso_x = (left_shoulder.x + right_shoulder.x) * w / 2
        torso_y = (left_shoulder.y + right_shoulder.y) * h / 2

        # Render the 2D dress image
        render_dress(frame, dress_image, torso_x, torso_y)

        # Removed the green rectangle placeholder since we're now rendering the dress image

    # Display the frame
    cv2.imshow("AR Dress Try-On", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()