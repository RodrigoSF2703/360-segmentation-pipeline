import os
import time
import cv2
import numpy as np

from src.evaluation.onnx_runner import ONNXSegmenter
from src.evaluation.metrics import compute_mean_iou


DATASET_DIR = "data/processed/yolo_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images/val")
MASKS_DIR = "data/processed/masks"
ASSETS_DIR = "assets"
N_EXAMPLES = 3


def find_onnx_model():
    for root, _, files in os.walk("runs"):
        for f in files:
            if f.endswith(".onnx"):
                return os.path.join(root, f)
    raise FileNotFoundError("Nenhum .onnx encontrado em runs/")


def save_overlay(image, mask, output_path, max_width_per_side=1280):
    """
    Salva original (esquerda) + overlay vermelho semi-transparente (direita)
    lado-a-lado em JPG. Reduz cada lado para max_width_per_side (padrão 1280),
    porque o original 5760x2880 fica pesado pra ir no repo.
    """
    h, w = image.shape[:2]

    if w > max_width_per_side:
        scale = max_width_per_side / w
        new_size = (max_width_per_side, int(h * scale))
        image = cv2.resize(image, new_size)
        mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

    red = np.zeros_like(image)
    red[mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(image, 0.7, red, 0.3, 0)

    side_by_side = np.hstack([image, blended])
    cv2.imwrite(output_path, side_by_side, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    model_path = find_onnx_model()
    print(f"Usando ONNX: {model_path}")

    segmenter = ONNXSegmenter(model_path)

    image_files = sorted(os.listdir(IMAGES_DIR))
    print(f"{len(image_files)} imagens em val/")

    os.makedirs(ASSETS_DIR, exist_ok=True)

    preds = []
    gts = []
    times = []

    # 🔥 warmup (descarta a primeira inferência do FPS)
    dummy = cv2.imread(os.path.join(IMAGES_DIR, image_files[0]))
    segmenter.predict(dummy)

    for i, file in enumerate(image_files):
        img_path = os.path.join(IMAGES_DIR, file)
        mask_path = os.path.join(MASKS_DIR, file)

        image = cv2.imread(img_path)
        gt = cv2.imread(mask_path, 0)

        if image is None or gt is None:
            continue

        start = time.time()
        pred = segmenter.predict(image)
        end = time.time()

        times.append(end - start)
        preds.append(pred)
        gts.append(gt)

        if i < N_EXAMPLES:
            out = os.path.join(ASSETS_DIR, f"example_{i + 1}.jpg")
            save_overlay(image, pred, out)

    avg_time = float(np.mean(times))
    fps = 1.0 / avg_time
    mean_iou = compute_mean_iou(preds, gts)

    print("\n=== RESULTADOS ===")
    print(f"Imagens avaliadas: {len(preds)}")
    print(f"IoU médio: {mean_iou:.4f}")
    print(f"Tempo médio: {avg_time * 1000:.1f} ms/imagem")
    print(f"FPS: {fps:.2f}")
    print(f"\nExemplos salvos em {ASSETS_DIR}/")


if __name__ == "__main__":
    main()
