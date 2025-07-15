import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (keeping the yolov8n.pt as requested)
model = YOLO('yolov8n.pt')

# Define COCO class labels, where 'person' is the first class
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Initialize video capture from file or camera
video_path = 'crowd_video.mp4'  # Replace with 0 for webcam
cap = cv2.VideoCapture(video_path)

# Confidence threshold to catch more detections (lowered for better sensitivity)
CONFIDENCE_THRESHOLD = 0.10
# Lower IoU threshold to ensure NMS works effectively with overlapping boxes
IOU_THRESHOLD = 0.6

# Maximum possible crowd count (for ratio calculation)
MAX_PEOPLE_COUNT = 350  # Set an appropriate value depending on the scenario

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection on the frame with NMS IoU threshold adjustments
    results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)

    # Extract bounding boxes, class IDs, and confidence scores
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Count the number of people in the frame
    current_people_count = sum(1 for i, class_id in enumerate(class_ids)
                               if COCO_CLASSES[class_id] == 'person' and confidences[i] > CONFIDENCE_THRESHOLD)

    # Calculate the crowd percentage
    crowd_percentage = (current_people_count / MAX_PEOPLE_COUNT) * 100 if MAX_PEOPLE_COUNT > 0 else 0

    # Draw bounding boxes for each detected person
    for i, class_id in enumerate(class_ids):
        if COCO_CLASSES[class_id] == 'person' and confidences[i] > CONFIDENCE_THRESHOLD:
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = map(int, boxes[i])
            # Draw bounding box and label for the person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person: {confidences[i]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the current people count and crowd percentage on the video
    text = f'Crowd Count: {current_people_count} | Crowd Traffic: {crowd_percentage:.2f}%'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color = (0, 0, 0)  # Black text
    background_color = (255, 255, 255)  # White background

    # Get the text size
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Coordinates for the text background box
    # Shift the text a bit to the left0
    text_x = 50  # Adjust for left position
    text_y = 30  # Position at the top of the frame

    # Draw the white rectangle as the background for the text
    cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10),
                  (text_x + text_size[0] + 10, text_y + 10), background_color, cv2.FILLED)

    # Draw the text over the rectangle
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    # Show the output frame
    cv2.imshow('Human Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()