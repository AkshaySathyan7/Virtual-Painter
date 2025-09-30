import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ---------- Settings ----------
CAM_ID = 0
BRUSH_THICKNESS = 7
SAVE_DIR = "saved_drawings"
# ------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Define color palette (BGR format)
colors = [
    (0, 0, 255),     # Red
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (255, 255, 0),   # Cyan
    (0, 0, 0),       # Black
    (255, 255, 255)  # White (acts like eraser)
]

color_names = ["RED", "GREEN", "BLUE", "YELLOW", "MAGENTA", "CYAN", "BLACK", "WHITE"]

# default color
color = colors[2]  # Blue
color_name = "BLUE"

# Prepare canvas and save dir
cap = cv2.VideoCapture(CAM_ID)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not open webcam")
h, w = frame.shape[:2]
canvas = np.zeros((h, w, 3), dtype=np.uint8)

os.makedirs(SAVE_DIR, exist_ok=True)

prev_x, prev_y = None, None

def draw_palette(img):
    """Draws the color palette on top of the screen"""
    box_size = 60
    margin = 10
    for i, c in enumerate(colors):
        x1 = margin + i * (box_size + margin)
        y1 = margin
        x2 = x1 + box_size
        y2 = y1 + box_size
        cv2.rectangle(img, (x1, y1), (x2, y2), c, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return box_size, margin

print("Controls:")
print(" - Draw: raise only index finger (point).")
print(" - Change color: move fingertip into a color box at the top row.")
print(" - Fist (all fingers down) = Clear canvas.")
print(" - Save: Press 's'. Quit: Press 'Esc'.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_text = ""
    drawing = False

    box_size, margin = draw_palette(frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        lm = hand_landmarks.landmark
        def lm_to_px(idx):
            return int(lm[idx].x * w), int(lm[idx].y * h)

        # fingers up/down detection
        tip_ids = [8, 12, 16, 20]
        fingers = []
        for tip in tip_ids:
            if lm[tip].y < lm[tip-2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        index_x, index_y = lm_to_px(8)

        # Check if fingertip is in palette area
        if index_y < box_size + margin:
            for i, c in enumerate(colors):
                x1 = margin + i * (box_size + margin)
                y1 = margin
                x2 = x1 + box_size
                y2 = y1 + box_size
                if x1 < index_x < x2 and y1 < index_y < y2:
                    color = c
                    color_name = color_names[i]
                    gesture_text = f"COLOR: {color_name}"

        # Drawing gesture: only index up
        elif fingers == [1, 0, 0, 0]:
            drawing = True
            gesture_text = f"DRAWING ({color_name})"
            if prev_x is None and prev_y is None:
                prev_x, prev_y = index_x, index_y
            cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), color, BRUSH_THICKNESS)
            prev_x, prev_y = index_x, index_y
        # Fist = clear
        elif fingers == [0, 0, 0, 0]:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            prev_x, prev_y = None, None
            gesture_text = "CLEAR CANVAS"
        else:
            prev_x, prev_y = None, None

    # Merge frame + canvas
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display gesture info
    cv2.rectangle(combined, (0, h-40), (400, h), (0, 0, 0), -1)
    cv2.putText(combined, f"{gesture_text}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Virtual Painter with Palette", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        timestamp = int(time.time())
        save_path = os.path.join(SAVE_DIR, f"drawing_{timestamp}.png")
        bg = np.full((h, w, 3), 255, dtype=np.uint8)
        saved_img = cv2.addWeighted(bg, 1.0, canvas, 1.0, 0)
        cv2.imwrite(save_path, saved_img)
        print(f"Saved drawing: {save_path}")

cap.release()
cv2.destroyAllWindows()
