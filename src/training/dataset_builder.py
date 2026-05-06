import os
import cv2
import shutil
import random
import numpy as np
from tqdm import tqdm


class YOLODatasetBuilder:
    def __init__(
        self,
        images_dir,
        masks_dir,
        output_dir,
        train_ratio=0.8,
        min_contour_area=200,
        seed=42,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.min_contour_area = min_contour_area

        random.seed(seed)

        self.train_img_dir = os.path.join(output_dir, "images/train")
        self.val_img_dir = os.path.join(output_dir, "images/val")
        self.train_lbl_dir = os.path.join(output_dir, "labels/train")
        self.val_lbl_dir = os.path.join(output_dir, "labels/val")

        for d in [
            self.train_img_dir,
            self.val_img_dir,
            self.train_lbl_dir,
            self.val_lbl_dir,
        ]:
            os.makedirs(d, exist_ok=True)

    def build(self):
        files = [f for f in os.listdir(self.images_dir) if f.endswith(".png")]
        random.shuffle(files)

        split_idx = int(len(files) * self.train_ratio)
        train_files = files[:split_idx]
        val_files = files[split_idx:]

        print(f"Train: {len(train_files)} | Val: {len(val_files)}")

        self._process_split(train_files, "train")
        self._process_split(val_files, "val")

        self._create_data_yaml()

    def _process_split(self, files, split):
        for file in tqdm(files, desc=split):
            img_path = os.path.join(self.images_dir, file)
            mask_path = os.path.join(self.masks_dir, file)

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)

            if image is None or mask is None:
                continue

            h, w = mask.shape

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            polygons = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_contour_area:
                    continue

                # simplificação leve
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                poly = []
                for point in approx:
                    x, y = point[0]
                    poly.append(x / w)
                    poly.append(y / h)

                if len(poly) >= 6:  # mínimo 3 pontos
                    polygons.append(poly)

            if not polygons:
                continue

            # salvar imagem
            if split == "train":
                out_img = os.path.join(self.train_img_dir, file)
                out_lbl = os.path.join(self.train_lbl_dir, file.replace(".png", ".txt"))
            else:
                out_img = os.path.join(self.val_img_dir, file)
                out_lbl = os.path.join(self.val_lbl_dir, file.replace(".png", ".txt"))

            shutil.copy(img_path, out_img)

            with open(out_lbl, "w") as f:
                for poly in polygons:
                    line = "0 " + " ".join([f"{p:.6f}" for p in poly])
                    f.write(line + "\n")

    def _create_data_yaml(self):
        yaml_path = os.path.join(self.output_dir, "data.yaml")

        # Path absoluto: o ultralytics resolve train/val a partir daqui.
        abs_path = os.path.abspath(self.output_dir)

        content = f"""
path: {abs_path}
train: images/train
val: images/val

names:
  0: motorcycle_person
"""

        with open(yaml_path, "w") as f:
            f.write(content.strip())