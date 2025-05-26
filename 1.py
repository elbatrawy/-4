from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from lane_detection import LaneDetection
from depth_module import estimate_depth, calculate_distance
from PIL import Image
import io

app = Flask(__name__)

# تحميل النماذج
model = YOLO("yolov5s.pt")
lane_detector = LaneDetection()

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame = np.array(pil_image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # تغيير الحجم
    frame_resized = cv2.resize(frame, (640, 360))

    # YOLO
    yolo_results = model(frame_resized)
    annotated_frame = yolo_results[0].plot()

    # كشف الحارات
    lane_input = cv2.resize(frame_resized, (512, 256)) / 255.0
    lane_input = np.rollaxis(lane_input, axis=2, start=0)
    lane_input = np.array([lane_input])
    _, _, lane_images = lane_detector.detect(lane_input)
    lane_mask = lane_images[0]

    # تقدير العمق
    distance_info = []
    depth_map = estimate_depth(frame_resized)

    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            class_name = result.names[int(box.cls[0])]
            distance = calculate_distance(depth_map, (x1, y1, x2, y2, conf))

            distance_info.append({
                "class": class_name,
                "confidence": round(conf, 2),
                "distance_m": round(distance, 2),
                "box": [x1, y1, x2, y2]
            })

    return jsonify({"detections": distance_info})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
