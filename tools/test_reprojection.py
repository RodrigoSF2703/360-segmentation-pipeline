import cv2
import json
import numpy as np
import os

from src.reprojection.reprojector import Reprojector


BASE_DIR = "data/intermediate/views"

# Escolhe uma pasta (uma imagem 360)
sample_folder = os.listdir(BASE_DIR)[0]
sample_path = os.path.join(BASE_DIR, sample_folder)

files = os.listdir(sample_path)

# Carregar metadata de qualquer arquivo (para pegar source_image)
json_file = [f for f in files if f.endswith(".json")][0]
with open(os.path.join(sample_path, json_file), "r") as f:
    metadata = json.load(f)

# Carregar imagem 360 original
orig_path = os.path.join("data/raw/training_images", metadata["source_image"])
orig = cv2.imread(orig_path)

H, W = orig.shape[:2]

# Inicializar reprojector
reprojector = Reprojector((H, W))

final_canvas = np.zeros((H, W), dtype=np.uint8)

# 🔁 Loop nas vistas
for file in files:
    if not file.endswith(".png"):
        continue

    img_file = file
    json_file = file.replace(".png", ".json")

    img_path = os.path.join(sample_path, img_file)
    json_path = os.path.join(sample_path, json_file)

    img = cv2.imread(img_path)

    if img is None:
        continue

    # 🧪 máscara fake
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), 100, 255, -1)

    with open(json_path, "r") as f:
        metadata = json.load(f)

    canvas = reprojector.reproject_mask(mask, metadata)

    # 🔥 fusão
    final_canvas = np.maximum(final_canvas, canvas)

# 💾 salvar máscara
cv2.imwrite("reprojected_fused_mask.png", final_canvas)

# 🎨 overlay
overlay = orig.copy()
overlay[final_canvas > 0] = [0, 0, 255]

cv2.imwrite("reprojected_fused_overlay.png", overlay)

print("Reprojeção com fusão concluída.")