import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define static triple concentric circle
center = (320, 240)
radii = [50, 100, 150]
thickness = 2
circle_color = (0, 255, 0)


# Initialize hand tracking module
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Open video capture device
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the image from BGR to RGB and pass to Mediapipe hands module
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    # Draw triple concentric circle at the center of the frame
    cv2.circle(frame, center, radii[0], circle_color, thickness)
    cv2.circle(frame, center, radii[1], circle_color, thickness)
    cv2.circle(frame, center, radii[2], circle_color, thickness)
    
    # Check if hand is detected in the frame
    if results.multi_hand_landmarks:
        # Get the landmarks for the first hand detected
        landmarks = results.multi_hand_landmarks[0].landmark
        
        # Extract the index finger coordinates
        index_finger_coordinates = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]), 
                                    int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]))
        
        

        # Draw a small circle around the index finger
        cv2.circle(frame, index_finger_coordinates, 10, (255, 0, 0), -1)
    
    # Show the frame with the drawn circle
    cv2.imshow('Hand Tracking', frame)
    
    # Check for keyboard input to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and destroy all windows
cap.release()
cv2.destroyAllWindows()
