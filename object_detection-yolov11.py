from ultralytics import YOLO
import cv2

# تحميل النموذج
model = YOLO("yolo11n.pt")

# تحميل الفيديو
video_path = r"C:\Users\user1\Downloads\carla - Made with Clipchamp\input_video.mp4"
cap = cv2.VideoCapture(video_path)

# إعداد كاتب الفيديو
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # التنبؤ على الفريم
    results = model(frame)
    annotated_frame = results[0].plot()

    # عرض الفريم
    cv2.imshow("YOLO11 Video Detection", annotated_frame)

    # حفظ الفريم للفيديو
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ saved: output_video.mp4")
