import cv2
import numpy as np
import os
from ultralytics import YOLO
from lane_detection import LaneDetection
from depth_module import estimate_depth  # تأكد إن estimate_depth موجود في ملف منفصل أو نفس الملف
from depth_module import calculate_distance
from datetime import datetime

# تحميل YOLO v5 (اختر نموذج أسرع مثل yolov5s)
model = YOLO("yolov5s.pt")

# تحميل كاشف المسارات
lane_detector = LaneDetection()

# تحميل الفيديو
video_path = r"C:\Users\user1\Downloads\carla - Made with Clipchamp\input_video.mp4"
cap = cv2.VideoCapture(video_path)

# إعداد إخراج الفيديو
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)  # تقليل الحجم إلى النصف
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)

# إنشاء مجلد لحفظ الفيديوهات
output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

# اسم الملف بتاريخ ووقت التشغيل
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file_main = os.path.join(output_dir, f"output_{current_time}.mp4")
output_file_depth = os.path.join(output_dir, f"output_{current_time}_depth.mp4")

# VideoWriters
out_main = cv2.VideoWriter(output_file_main, fourcc, fps, (width, height))
out_depth = cv2.VideoWriter(output_file_depth, fourcc, fps, (width, height))

frame_skip = 2  # تخطي كل 2 إطار لتسريع المعالجة
frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_counter % frame_skip == 0:
        # تقليل الحجم إلى النصف
        frame_resized = cv2.resize(frame, (width, height))

        # ----------------- YOLO Object Detection -----------------
        yolo_results = model(frame_resized)
        annotated_frame = yolo_results[0].plot()

        # ----------------- Lane Detection -----------------
        lane_input = cv2.resize(frame_resized, (512, 256)) / 255.0
        lane_input = np.rollaxis(lane_input, axis=2, start=0)
        lane_input = np.array([lane_input])
        _, _, lane_images = lane_detector.detect(lane_input)
        lane_mask = lane_images[0]

        # تجهيز الماسك والدمج
        mask_gray = cv2.cvtColor(lane_mask, cv2.COLOR_RGB2GRAY)
        mask_gray = np.where(mask_gray > 0, 255, mask_gray).astype(np.uint8)
        mask_inv = cv2.bitwise_not(mask_gray)

        if mask_inv.shape != annotated_frame.shape[:2]:
            mask_inv = cv2.resize(mask_inv, (annotated_frame.shape[1], annotated_frame.shape[0]))

        image_masked = cv2.bitwise_and(annotated_frame, annotated_frame, mask=mask_inv)

        if lane_mask.shape[:2] != annotated_frame.shape[:2]:
            lane_mask = cv2.resize(lane_mask, (annotated_frame.shape[1], annotated_frame.shape[0]))

        combined_frame = cv2.add(image_masked, lane_mask)

        # استخراج الكائنات من نتائج YOLO
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # تحويل الإحداثيات إلى أرقام صحيحة
                conf = float(box.conf[0])  # استخراج الاحتمال (الثقة)
                class_name = result.names[int(box.cls[0])]  # استخراج اسم الكائن

                # ----------------- Depth Estimation -----------------
                # هنا يتم تقدير العمق باستخدام خريطة العمق لكل إطار
                depth_map = estimate_depth(frame_resized)  # حساب خريطة العمق للإطار الحالي
                distance = calculate_distance(depth_map, (x1, y1, x2, y2, conf))  # حساب المسافة

                # رسم المربع المحيط بالكائن مع المسافة
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # وضع المسافة تحت المربع (إما يسار المربع أو يمين المربع)
                text_position = (x1 + 5, y2 + 15)  # المسافة ستكون أسفل المربع على الجهة اليسرى
                cv2.putText(combined_frame, f"{distance:.2f}m", text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # عرض المسافة

                # عرض اسم الكائن (اختياري)
                #cv2.putText(combined_frame, class_name, (x1, y1 - 5),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # عرض اسم الكائن

        # تحويل خريطة العمق إلى صورة مرئية
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_vis_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        # إظهار الفيديو مع الكائنات والمسافة
        cv2.imshow("Objects with Distance", combined_frame)
        out_main.write(combined_frame)  # حفظ الفيديو مع الكشف والمسافة
        out_depth.write(depth_vis_colored)  # حفظ الفيديو الذي يحتوي على خريطة العمق

    frame_counter += 1

    # التحقق من الضغط على المفتاح للخروج
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # للخروج
        break

# تحرير الموارد
cap.release()
out_main.release()
out_depth.release()
cv2.destroyAllWindows()

# طباعة مسارات الفيديوهات المحفوظة
print(f"✅ Main video saved: {output_file_main}")
print(f"✅ Depth video saved: {output_file_depth}")



