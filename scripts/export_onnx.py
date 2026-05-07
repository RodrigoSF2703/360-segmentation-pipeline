import os
from ultralytics import YOLO


RUNS_DIR = "runs"


def find_latest_best_model():
    candidates = []

    for root, _, files in os.walk(RUNS_DIR):
        for f in files:
            if f == "best.pt":
                full_path = os.path.join(root, f)
                candidates.append(full_path)

    if not candidates:
        raise FileNotFoundError("Nenhum best.pt encontrado em runs/")

    # pega o mais recente
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def main():
    best_model_path = find_latest_best_model()
    print(f"Usando modelo: {best_model_path}")

    model = YOLO(best_model_path)

    print("Exportando para ONNX...")
    model.export(format="onnx")

    print("✔️ Export concluído")


if __name__ == "__main__":
    main()