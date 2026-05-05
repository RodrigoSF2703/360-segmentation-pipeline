import os
import cv2
import yaml
from tqdm import tqdm

from src.extraction.extractor import ViewExtractor


INPUT_DIR = "data/raw/training_images"
OUTPUT_DIR = "data/intermediate/views"


def load_config():
    with open("configs/pipeline.yaml", "r") as f:
        return yaml.safe_load(f)


def extract_views(extractor, image, config):
    """
    Gera as vistas planas conforme o modo definido no config.
    Retorna a lista de vistas (dicts com imagem + metadados).
    """
    mode = config["mode"]

    if mode == "single":
        cfg = config["single"]
        return extractor.extract_multiple_views(
            image=image,
            yaw_list=[cfg["yaw"]],
            pitch=cfg["pitch"],
        )

    if mode == "nadir_multi":
        cfg = config["nadir_multi"]
        return extractor.extract_multiple_views(
            image=image,
            yaw_list=cfg["yaw_list"],
            pitch=cfg["pitch"],
        )

    if mode == "grid":
        cfg = config["grid"]
        yaw_list = list(range(0, 360, cfg["yaw_step"]))
        # Pula os polos exatos (-90 e +90) — perspectiva degenerada.
        pitch_list = list(range(-90 + cfg["pitch_step"], 90, cfg["pitch_step"]))
        return extractor.extract_grid_views(
            image=image,
            yaw_list=yaw_list,
            pitch_list=pitch_list,
        )

    raise ValueError(f"Modo inválido: {mode}")


def main():
    config = load_config()

    print(f"Modo de extração: {config['mode']}")

    extractor = ViewExtractor(
        fov=config["fov"],
        output_size=tuple(config["output_size"]),
    )

    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ]
    print(f"{len(image_files)} imagens encontradas.")

    for img_name in tqdm(image_files):
        img_path = os.path.join(INPUT_DIR, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Erro ao carregar: {img_name}")
            continue

        base_name = os.path.splitext(img_name)[0]
        image_output_dir = os.path.join(OUTPUT_DIR, base_name)

        views = extract_views(extractor, image, config)

        extractor.save_views(
            views=views,
            output_dir=image_output_dir,
            base_name=base_name,
            source_image=img_name,
        )

    print("Processamento concluído.")


if __name__ == "__main__":
    main()
