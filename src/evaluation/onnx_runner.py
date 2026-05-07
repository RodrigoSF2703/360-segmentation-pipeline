import cv2
import numpy as np
from ultralytics import YOLO


class ONNXSegmenter:
    """
    Inferência da YOLOv8-seg em CPU a partir do .onnx exportado.

    Carrega via wrapper do ultralytics, que usa onnxruntime com
    CPUExecutionProvider quando device="cpu". O pós-processamento (NMS,
    prototypes -> instance masks) é feito pelo próprio ultralytics.

    A predição usa result.masks.xy (polígonos já em coords da imagem original,
    com letterbox desfeito) em vez de result.masks.data, que retorna no
    espaço 640x640 letterboxado do modelo.
    """

    def __init__(self, model_path, imgsz=640):
        self.model = YOLO(model_path, task="segment")
        self.imgsz = imgsz

    def predict(self, image):
        """
        image: np.ndarray BGR (H, W, 3).
        Retorna máscara binária (H, W) uint8 com a união de todas as
        instâncias detectadas (single-class motorcycle_person).
        """
        results = self.model.predict(
            image,
            device="cpu",
            imgsz=self.imgsz,
            verbose=False,
        )

        result = results[0]
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if result.masks is None:
            return mask

        for poly in result.masks.xy:
            if len(poly) < 3:
                continue
            cv2.fillPoly(mask, [poly.astype(np.int32)], 255)

        return mask
