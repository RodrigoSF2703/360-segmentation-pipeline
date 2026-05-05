import cv2
import os
import numpy as np
from PIL import Image
from lang_sam import LangSAM

# =========================
# 1. CARREGAR MODELO
# =========================
print("Carregando modelo LangSAM...")
model = LangSAM()

# =========================
# 2. PEGAR UMA VISTA
# =========================
BASE_DIR = "data/intermediate/views"

sample_folder = os.listdir(BASE_DIR)[0]
sample_path = os.path.join(BASE_DIR, sample_folder)

image_files = [f for f in os.listdir(sample_path) if f.endswith(".png")]

if not image_files:
    raise Exception("Nenhuma imagem encontrada na pasta de views.")

image_file = image_files[0]
image_path = os.path.join(sample_path, image_file)

print(f"Usando imagem: {image_path}")

# =========================
# 3. CARREGAR IMAGEM
# =========================
image = cv2.imread(image_path)

if image is None:
    raise Exception(f"Erro ao carregar imagem: {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)

# =========================
# 4. PROMPT
# =========================
text_prompt = "person on a motorcycle"

# =========================
# 5. INFERÊNCIA
# =========================
# API atual do lang_sam espera listas e retorna lista de dicts.
print("Rodando LangSAM...")
results = model.predict([image_pil], [text_prompt])

result = results[0]
masks = result["masks"]
boxes = result["boxes"]
scores = result["scores"]
labels = result["labels"]

print(f"Detectou {len(masks)} máscaras")

# =========================
# 6. CRIAR MÁSCARA FINAL
# =========================
final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

for mask in masks:
    final_mask[mask.astype(bool)] = 255

# =========================
# 7. SALVAR RESULTADOS
# =========================
cv2.imwrite("langsam_mask.png", final_mask)

overlay = image.copy()
overlay[final_mask > 0] = [0, 0, 255]

cv2.imwrite("langsam_overlay.png", overlay)

print("Resultado salvo:")
print("- langsam_mask.png")
print("- langsam_overlay.png")