# depth_module.py

import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

# إعداد MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# تحويل الصور لـ MiDaS
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

# === لحساب المسافة باستخدام خريطة العمق ===
KNOWN_CAR_WIDTH = 1.8  # بالمتر
KNOWN_DISTANCE = 10.0
KNOWN_PIXEL_WIDTH = 300

FOCAL_LENGTH = (KNOWN_PIXEL_WIDTH * KNOWN_DISTANCE) / KNOWN_CAR_WIDTH

def calculate_distance(depth_map, bbox):
    x1, y1, x2, y2, _ = bbox
    car_depth = depth_map[y1:y2, x1:x2]
    avg_depth = np.mean(car_depth)
    pixel_width = x2 - x1
    if pixel_width == 0:
        return 0.0
    return (FOCAL_LENGTH * KNOWN_CAR_WIDTH) / pixel_width
