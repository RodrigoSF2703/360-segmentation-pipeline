import os
import cv2
import yaml
from tqdm import tqdm

from src.extraction.extractor import ViewExtractor


INPUT_DIR = "data/raw/training_images"
OUTPUT_DIR = "data/intermediate/views"


def main():
    # 🔧 Carregar config
    with open("configs/pipeline.yaml", "r") as f:
        config = yaml.safe_load(f)

    cfg = config["extraction"]

    print(f"Modo de extração: {cfg['mode']}")
    
    # Criar extractor
    extractor = ViewExtractor(
        fov=cfg["fov"],
        output_size=tuple(cfg["output_size"])
    )

    # Listar imagens
    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ]

    print(f"{len(image_files)} imagens encontradas.")

    # Loop principal
    for img_name in tqdm(image_files):
        img_path = os.path.join(INPUT_DIR, img_name)

        image = cv2.imread(img_path)

        if image is None:
            print(f"Erro ao carregar: {img_name}")
            continue

        base_name = os.path.splitext(img_name)[0]

        # Pasta específica da imagem
        image_output_dir = os.path.join(OUTPUT_DIR, base_name)

        # 🔥 Seleção de modo
        if cfg["mode"] == "nadir":
            views = extractor.extract_multiple_views(
                image=image,
                yaw_list=cfg["nadir"]["yaw_list"],
                pitch=cfg["nadir"]["pitch"]
            )

        elif cfg["mode"] == "grid":
            views = extractor.extract_grid_views(
                image=image,
                yaw_list=cfg["grid"]["yaw_list"],
                pitch_list=cfg["grid"]["pitch_list"]
            )

        else:
            raise ValueError(f"Modo inválido: {cfg['mode']}")

        print(f"{img_name}: {len(views)} views geradas")
        
        # Salvar
        extractor.save_views(
            views=views,
            output_dir=image_output_dir,
            base_name=base_name,
            source_image=img_name
        )

    print("Processamento concluído.")


if __name__ == "__main__":
    main()