"""
Diagnóstico visual: sobrepõe GT (verde) e predição ONNX (vermelho) na
mesma imagem para checar se o modelo está prevendo no lugar certo.

Saída em assets/diagnose/, 3 painéis lado a lado por imagem:
  Original | GT (verde) | Predição (vermelho)
"""

import os
import cv2
import numpy as np

from src.evaluation.onnx_runner import ONNXSegmenter


IMAGES_DIR = "data/processed/yolo_dataset/images/val"
MASKS_DIR = "data/processed/masks"
OUT_DIR = "assets/diagnose"
N_SAMPLES = 3
MAX_WIDTH_PER_PANEL = 900


def find_onnx_model():
    for root, _, files in os.walk("runs"):
        for f in files:
            if f.endswith(".onnx"):
                return os.path.join(root, f)
    raise FileNotFoundError("Nenhum .onnx encontrado em runs/")


def overlay(image, mask, color):
    canvas = np.zeros_like(image)
    canvas[mask > 0] = color
    return cv2.addWeighted(image, 0.6, canvas, 0.4, 0)


def shrink(img, target_w):
    h, w = img.shape[:2]
    if w <= target_w:
        return img
    scale = target_w / w
    return cv2.resize(img, (target_w, int(h * scale)))


def main():
    model_path = find_onnx_model()
    print(f"Usando ONNX: {model_path}")

    segmenter = ONNXSegmenter(model_path)
    os.makedirs(OUT_DIR, exist_ok=True)

    image_files = sorted(os.listdir(IMAGES_DIR))[:N_SAMPLES]

    for i, file in enumerate(image_files):
        img_path = os.path.join(IMAGES_DIR, file)
        mask_path = os.path.join(MASKS_DIR, file)

        image = cv2.imread(img_path)
        gt = cv2.imread(mask_path, 0)

        if image is None or gt is None:
            print(f"Pulando {file}: imagem ou GT ausente")
            continue

        gt = np.squeeze(gt)
        pred = segmenter.predict(image)

        gt_overlay = overlay(image, gt, (0, 255, 0))     # verde
        pred_overlay = overlay(image, pred, (0, 0, 255)) # vermelho

        # Reduz cada painel antes de empilhar (originais 5760x2880 são grandes)
        a = shrink(image, MAX_WIDTH_PER_PANEL)
        b = shrink(gt_overlay, MAX_WIDTH_PER_PANEL)
        c = shrink(pred_overlay, MAX_WIDTH_PER_PANEL)

        side = np.hstack([a, b, c])
        out = os.path.join(OUT_DIR, f"diagnose_{i + 1}.jpg")
        cv2.imwrite(out, side, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"Salvo: {out}")

    print(f"\nResultado em {OUT_DIR}/")


if __name__ == "__main__":
    main()
