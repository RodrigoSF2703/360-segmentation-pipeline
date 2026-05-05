import numpy as np
from PIL import Image
from lang_sam import LangSAM


class LangSAMSegmenter:
    def __init__(
        self,
        prompt="person on a motorcycle",
        box_threshold=0.3,
        text_threshold=0.25,
    ):
        self.model = LangSAM()
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def segment(self, image_bgr):
        """
        Recebe imagem (BGR - OpenCV)
        Retorna máscara binária (uint8)
        """

        # BGR → RGB
        image_rgb = image_bgr[:, :, ::-1]

        # evitar warning PyTorch
        image_pil = Image.fromarray(image_rgb.copy())

        try:
            results = self.model.predict(
                [image_pil],
                [self.prompt],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )

            result = results[0]
            masks = result.get("masks", [])

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
            return np.zeros(image_bgr.shape[:2], dtype=np.uint8)