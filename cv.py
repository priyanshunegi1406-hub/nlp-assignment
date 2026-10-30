import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
#WhatsApp Video 2026-10-30 at 10.50.12 PM.mp4
#  dance
input_video = "dance.mp4"
output_video = "coco_pose_black.mp4"

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error opening video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

while True:

    ret, frame = cap.read()

    if not ret:
        break

    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    results = model(frame, conf=0.25, verbose=False)

    for person in results[0].keypoints:

        points = person.xy[0].cpu().numpy()
        confidences = person.conf[0].cpu().numpy()

        for p1, p2 in skeleton:

            if confidences[p1] > 0.25 and confidences[p2] > 0.25:

                x1, y1 = map(int, points[p1])
                x2, y2 = map(int, points[p2])

                cv2.line(
                    black_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

        for i, (x, y) in enumerate(points):

            if confidences[i] > 0.25:

                cv2.circle(
                    black_frame,
                    (int(x), int(y)),
                    4,
                    (255, 255, 255),
                    -1
                )

    out.write(black_frame)

cap.release()
out.release()

cv2.destroyAllWindows()

print("Done! Output saved as", output_video)