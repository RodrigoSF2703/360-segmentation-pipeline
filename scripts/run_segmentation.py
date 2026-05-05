import os
import cv2
import json
import numpy as np
from tqdm import tqdm

from src.segmentation.langsam_segmenter import LangSAMSegmenter
from src.reprojection.reprojector import Reprojector

BASE_VIEWS_DIR = "data/intermediate/views"
BASE_RAW_DIR = "data/raw/training_images"

OUTPUT_IMG_DIR = "data/processed/images"
OUTPUT_MASK_DIR = "data/processed/masks"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# 🔥 CONTROLE (reduzido para teste)
MAX_IMAGES = 20
MAX_VIEWS = 15

def main():
    segmenter = LangSAMSegmenter()

    folders = os.listdir(BASE_VIEWS_DIR)[:MAX_IMAGES]

    for folder in folders:
        print(f"\nProcessando: {folder}")

        view_path = os.path.join(BASE_VIEWS_DIR, folder)
        files = [f for f in os.listdir(view_path) if f.endswith(".png")][:MAX_VIEWS]

        # carregar imagem original
        sample_json = files[0].replace(".png", ".json")
        with open(os.path.join(view_path, sample_json), "r") as f:
            metadata = json.load(f)

        orig_path = os.path.join(BASE_RAW_DIR, metadata["source_image"])
        orig = cv2.imread(orig_path)

        if orig is None:
            print(f"Erro ao carregar original: {orig_path}")
            continue

        H, W = orig.shape[:2]
        reprojector = Reprojector((H, W))

        canvases = []

        for file in tqdm(files, desc=folder):
            img_path = os.path.join(view_path, file)
            json_path = img_path.replace(".png", ".json")

            image = cv2.imread(img_path)
            if image is None:
                continue

            # 🔥 SEGMENTAÇÃO
            try:
                mask = segmenter.segment(image)
            except Exception as e:
                print("[ERRO LangSAM]", e)
                continue

            if mask is None or mask.sum() == 0:
                continue

            # 🔥 METADATA
            with open(json_path, "r") as f:
                metadata = json.load(f)

            # 🔥 REPROJEÇÃO
            canvases.append(reprojector.reproject_mask(mask, metadata))

        # 🔥 FUSÃO POR VOTO (threshold=1 = união; aumentar exige overlap entre vistas)
        if canvases:
            final_mask = Reprojector.fuse_masks(canvases, threshold=1)
        else:
            final_mask = np.zeros((H, W), dtype=np.uint8)

        # 🔥 SALVAR RESULTADO FINAL
        out_img_path = os.path.join(OUTPUT_IMG_DIR, folder + ".png")
        out_mask_path = os.path.join(OUTPUT_MASK_DIR, folder + ".png")

        cv2.imwrite(out_img_path, orig)
        cv2.imwrite(out_mask_path, final_mask)

        print(f"✔️ Salvo: {folder}")

if __name__ == "__main__":
    main()