import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import os

# === إعدادات المعايرة ===
KNOWN_CAR_WIDTH = 1.8
KNOWN_DISTANCE = 10.0
KNOWN_PIXEL_WIDTH = 300
FOCAL_LENGTH = (KNOWN_PIXEL_WIDTH * KNOWN_DISTANCE) / KNOWN_CAR_WIDTH

# === تحميل YOLOv11 ===
yolo_model = YOLO("yolo11n.pt")

# === تحميل MiDaS ===
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def estimate_depth(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = midas(img)
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))
    return depth_map

def detect_cars(image):
    results = yolo_model(image)
    detections = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if int(cls) in [2, 7]:  # Car or Truck
                detections.append((int(x1), int(y1), int(x2), int(y2), conf))
    return detections

def calculate_distance(depth_map, bbox):
    x1, y1, x2, y2, _ = bbox
    car_depth = depth_map[y1:y2, x1:x2]
    avg_depth = np.mean(car_depth)
    pixel_width = x2 - x1
    if pixel_width == 0:
        return 0.0
    return (FOCAL_LENGTH * KNOWN_CAR_WIDTH) / pixel_width

def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    # ==== إعداد المجلد ====
    output_dir = os.path.abspath("output_videos")
    os.makedirs(output_dir, exist_ok=True)

    # إعداد الفيديوهات
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    filename = os.path.splitext(os.path.basename(video_source))[0]
    out_path = os.path.join(output_dir, f"{filename}_with_distance.mp4")
    depth_path = os.path.join(output_dir, f"{filename}_depth.mp4")

    out_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    out_depth = cv2.VideoWriter(depth_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        depth_map = estimate_depth(frame)
        detections = detect_cars(frame)

        # رسم العمق كـ heatmap
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_vis_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        # رسم الصناديق والمسافات
        for bbox in detections:
            x1, y1, x2, y2, conf = bbox
            distance = calculate_distance(depth_map, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Dist: {distance:.2f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out_video.write(frame)
        out_depth.write(depth_vis_colored)

        # عرض حي لو تحب
        cv2.imshow("Original Video", frame)
        cv2.imshow("Depth Map", depth_vis_colored)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out_video.release()
    out_depth.release()
    cv2.destroyAllWindows()

    print(f"✅ Saved video with distances: {out_path}")
    print(f"✅ Saved depth map video: {depth_path}")

# ========== تشغيل ==========

if __name__ == "__main__":
    #video_source = r"C:\Users\user1\Downloads\carla - Made with Clipchamp\input_video.mp4"
    video_source = r"C:\Users\user1\Downloads\project_video.mp4"
    process_video(video_source)



