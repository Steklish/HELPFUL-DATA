import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.predict(frame, conf=0.01, verbose=True)
    annotated = results[0].plot()

    cv2.imshow("YOLOv8", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()