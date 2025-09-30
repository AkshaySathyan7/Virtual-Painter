import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ---------- Settings ----------
CAM_ID = 0
BRUSH_THICKNESS = 7
ERASER_THICKNESS = 50
SAVE_DIR = "saved_drawings"
# ------------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Color palette (BGR)
tools = [
    {"color": (0, 0, 255), "name": "RED"},
    {"color": (0, 255, 0), "name": "GREEN"},
    {"color": (255, 0, 0), "name": "BLUE"},
    {"color": (0, 255, 255), "name": "YELLOW"},
    {"color": (255, 0, 255), "name": "MAGENTA"},
    {"color": (255, 255, 0), "name": "CYAN"},
    {"color": (0, 0, 0), "name": "BLACK"},
    {"color": (255, 255, 255), "name": "WHITE"},
    {"color": None, "name": "ERASER"},      # Special tool
    {"color": None, "name": "CLEAR ALL"}    # Special tool
]

# Default tool
current_tool = tools[2]  # BLUE
color = current_tool["color"]

# Prepare canvas
cap = cv2.VideoCapture(CAM_ID)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not open webcam")
h, w = frame.shape[:2]
canvas = np.zeros((h, w, 3), dtype=np.uint8)

os.makedirs(SAVE_DIR, exist_ok=True)
prev_x, prev_y = None, None

def draw_palette(img):
    """Draws palette with color boxes and names"""
    box_w, box_h = 80, 80
    margin = 15
    start_x = margin
    y1, y2 = margin, margin + box_h

    for i, tool in enumerate(tools):
        x1 = start_x + i * (box_w + margin)
        x2 = x1 + box_w

        # Special tools
        if tool["name"] == "ERASER":
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), -1)
            cv2.putText(img, "E", (x1+20, y1+55), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
        elif tool["name"] == "CLEAR ALL":
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.putText(img, "X", (x1+20, y1+55), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), tool["color"], -1)

        # Border highlight if selected
        if tool == current_tool:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 4)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        # Tool name
        cv2.putText(img, tool["name"], (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    return box_w, box_h, margin

print("Controls:")
print(" - Draw: index finger only (point).")
print(" - Select color/tool: move fingertip into a palette box.")
print(" - Eraser: removes strokes.")
print(" - CLEAR ALL: wipes full canvas.")
print(" - Save: Press 's'. Quit: Press 'Esc'.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_text = ""
    drawing = False

    box_w, box_h, margin = draw_palette(frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        lm = hand_landmarks.landmark
        def lm_to_px(idx):
            return int(lm[idx].x * w), int(lm[idx].y * h)

        # Detect finger up/down
        tip_ids = [8, 12, 16, 20]
        fingers = []
        for tip in tip_ids:
            fingers.append(1 if lm[tip].y < lm[tip-2].y else 0)

        index_x, index_y = lm_to_px(8)

        # Check palette selection
        if index_y < box_h + margin:
            for i, tool in enumerate(tools):
                x1 = margin + i * (box_w + margin)
                x2 = x1 + box_w
                y1, y2 = margin, margin + box_h
                if x1 < index_x < x2 and y1 < index_y < y2:
                    current_tool = tool
                    color = tool["color"]
                    gesture_text = f"TOOL: {tool['name']}"
                    if tool["name"] == "CLEAR ALL":
                        canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    prev_x, prev_y = None, None

        # Drawing mode: index finger only
        elif fingers == [1, 0, 0, 0]:
            if current_tool["name"] == "ERASER":
                cv2.circle(canvas, (index_x, index_y), ERASER_THICKNESS, (0,0,0), -1)
                gesture_text = "ERASING"
            else:
                if prev_x is None and prev_y is None:
                    prev_x, prev_y = index_x, index_y
                cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), color, BRUSH_THICKNESS)
                prev_x, prev_y = index_x, index_y
                gesture_text = f"DRAWING ({current_tool['name']})"
        else:
            prev_x, prev_y = None, None

    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.rectangle(combined, (0, h-40), (400, h), (0,0,0), -1)
    cv2.putText(combined, f"{gesture_text}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Virtual Painter Pro", combined)

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
