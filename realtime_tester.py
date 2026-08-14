from ultralytics import YOLO
import cv2
import cvzone
import math

# Change this line
model = YOLO(r"D:\DEVELOPMENT\All Projects\Park Prabandh - Realtime Parking Spot Detection & Vehicle Assigning\CRIST\park-prabandh\final_model.pt")

print("Detecting available webcams...")
available_cameras = []
for i in range(10):
    cap_test = cv2.VideoCapture(i)
    if cap_test.isOpened():
        available_cameras.append(i)
        cap_test.release()
    else:
        cap_test.release()

if not available_cameras:
    raise IOError("Error: No cameras found. Please check your webcam connection.")

print("\nAvailable cameras found:")
for idx, cam_index in enumerate(available_cameras):
    print(f"  {idx + 1}. Camera {cam_index}")

selected_camera_index = -1
while True:
    try:
        choice = input(f"Enter the number (1-{len(available_cameras)}) of the camera you want to use: ")
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(available_cameras):
            selected_camera_index = available_cameras[choice_idx]
            break
        else:
            print(f"Invalid choice. Please enter a number between 1 and {len(available_cameras)}.")
    except ValueError:
        print("Invalid input. Please enter a number.")

print(f"Attempting to use Camera {selected_camera_index}...")


cap = cv2.VideoCapture(selected_camera_index) # Use the selected camera index

if not cap.isOpened():
    raise IOError(f"Error: Cannot open selected camera {selected_camera_index}")

cap.set(3, 1920) # Set width
cap.set(4, 1080) # Set height

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Requesting 1920x1080... Actual feed resolution: {width}x{height}")

# Class order: 0='occupied_slot', 1='free_slot'
classNames = ['occupied_slot', 'free_slot']

print("Processing started... Press 'q' to quit.")

frame_count = 0

cv2.namedWindow("Real-time Detection", cv2.WINDOW_NORMAL)

while True:
    ret, img = cap.read()
    frame_count += 1

    if not ret or img is None:
        print(f"Error: Failed to grab frame {frame_count}. 'ret'={ret}, 'img' is None={img is None}")
        cv2.waitKey(10)
        continue  # Skip the rest of the loop and try the next frame

    if frame_count % 30 == 0:  # Print every 30 frames to avoid spamming console
        print(f"Processing frame {frame_count}...")

    results = model(img, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = math.ceil(box.conf[0] * 100) / 100

            if cls >= len(classNames):
                continue

            class_name = classNames[cls]

            # --- MODIFICATION START ---

            # 1. Define border color based on class name
            # Note: OpenCV uses (B, G, R) color format
            if class_name == 'occupied_slot':
                box_color = (0, 0, 255)  # Red
            elif class_name == 'free_slot':
                box_color = (0, 255, 0)  # Green
            else:
                box_color = (255, 255, 255) # White (fallback)

            # 2. Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # 3. Draw the border using the selected color
            # We set both the rectangle (colorC) and corners (colorR) to the same color
            cvzone.cornerRect(img, (x1, y1, w, h), colorC=box_color, colorR=box_color)

            # 4. The text label line (cvzone.putTextRect) has been removed.

            # --- MODIFICATION END ---

    cv2.imshow("Real-time Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✅ Video processing stopped.")