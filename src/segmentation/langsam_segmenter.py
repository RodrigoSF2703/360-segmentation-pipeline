import numpy as np
from PIL import Image
from lang_sam import LangSAM


class LangSAMSegmenter:
    def __init__(self, prompt="person on a motorcycle"):
        self.model = LangSAM()
        self.prompt = prompt

    def segment(self, image_bgr):
        """
        Recebe imagem (BGR - OpenCV)
        Retorna máscara binária (uint8)
        """

        # BGR → RGB
        image_rgb = image_bgr[:, :, ::-1]

        # IMPORTANTE: copiar para evitar warning do PyTorch
        image_pil = Image.fromarray(image_rgb.copy())

        try:
            results = self.model.predict([image_pil], [self.prompt])
            result = results[0]

            masks = result.get("masks", [])

            # fallback se não detectar nada
            if masks is None or len(masks) == 0:
                return np.zeros(image_bgr.shape[:2], dtype=np.uint8)

            final_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

            for mask in masks:
                if mask is None:
                    continue
                final_mask[mask.astype(bool)] = 255

            return final_mask

        except Exception as e:
            print(f"[ERRO LangSAM] {e}")
            # fallback total
            return np.zeros(image_bgr.shape[:2], dtype=np.uint8)