import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo26n-pose.pt")
video = cv2.VideoCapture("dance.mp4")

if not video.isOpened():
    print("Unable to open video")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30.0

output = cv2.VideoWriter(
    "coco_pose_black.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (w, h)
)

skeleton = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    black = np.zeros((h, w, 3), dtype=np.uint8)
    result = model(frame, conf=0.25, verbose=False)
    if result[0].keypoints is not None:
        for person in result[0].keypoints:
            points = person.xy[0].cpu().numpy()
            conf = person.conf[0].cpu().numpy()
            for a, b in skeleton:
                if conf[a] > 0.25 and conf[b] > 0.25:
                    x1, y1 = map(int, points[a])
                    x2, y2 = map(int, points[b])
                    cv2.line(black, (x1, y1), (x2, y2), (255, 255, 255), 2)
            for i in range(len(points)):
                if conf[i] > 0.25:
                    x, y = map(int, points[i])
                    cv2.circle(black, (x, y), 4, (255, 255, 255), -1)
    output.write(black)
    count += 1
    if count % 30 == 0:
        print("Frames processed:", count)

video.release()
output.release()
cv2.destroyAllWindows()
print("Video saved successfully!")
