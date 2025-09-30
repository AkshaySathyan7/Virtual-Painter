import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime

# Mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Create folder to save drawings
if not os.path.exists("saved_drawings"):
    os.makedirs("saved_drawings")

# Colors and Palette (only Red, Green, Blue, Clear)
palette = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Clear": (200, 200, 200)
}

# Palette coordinates (x1, y1, x2, y2)
palette_boxes = {
    "Red": (20, 20, 140, 80),
    "Green": (160, 20, 280, 80),
    "Blue": (300, 20, 420, 80),
    "Clear": (460, 20, 620, 80)
}

# Default brush settings
current_color = (0, 0, 255)
brush_thickness = 7
canvas = None

# Draw stylish palette with names inside
def draw_palette(frame):
    for name, (x1, y1, x2, y2) in palette_boxes.items():
        color = palette[name]

        # Filled rounded rectangle effect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)

        # Border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), 3, cv2.LINE_AA)

        # Text (centered inside box)
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(frame, name, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# Open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    finger_pos = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Index finger tip
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            finger_pos = (cx, cy)

            cv2.circle(frame, finger_pos, 8, (255, 255, 255), -1)

            # Check if finger touches palette
            for name, (x1, y1, x2, y2) in palette_boxes.items():
                if x1 < cx < x2 and y1 < cy < y2:
                    if name == "Clear":
                        canvas = np.zeros_like(frame)
                    else:
                        current_color = palette[name]

            # Otherwise, draw on canvas
            if cy > 100:  # avoid accidental palette touches
                cv2.circle(canvas, finger_pos, brush_thickness, current_color, -1)

    # Combine frame and canvas
    blended = cv2.addWeighted(frame, 0.5, canvas, 1, 0)

    # Draw palette on top
    draw_palette(blended)

    cv2.imshow("🎨 Virtual Painter Pro", blended)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord("s"):  # Save
        filename = f"saved_drawings/drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, canvas)
        print(f"Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
