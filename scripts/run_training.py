import yaml
from ultralytics import YOLO


def load_config():
    with open("configs/pipeline.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    cfg = config["training"]

    print(f"Carregando modelo base: {cfg['base_model']}")
    model = YOLO(cfg["base_model"])

    print(f"Iniciando treino em {cfg['device']}")
    model.train(
        data=cfg["data_yaml"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=cfg["device"],
        patience=cfg["patience"],
        project=cfg["project"],
        name=cfg["name"],
    )

    print("Treino concluído.")


if __name__ == "__main__":
    main()
