import cv2
import numpy as np


class MaskRefiner:
    def __init__(
        self,
        kernel_size=5,
        min_area=500,
        use_grabcut=False,
        grabcut_iter=5,
    ):
        self.kernel_size = kernel_size
        self.min_area = min_area
        self.use_grabcut = use_grabcut
        self.grabcut_iter = grabcut_iter

        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def refine(self, image, mask):
        """
        image: BGR (original)
        mask:  uint8 (0/255)
        """

        if mask is None or mask.sum() == 0:
            return mask

        # =========================
        # 1. OPEN (remover ruído)
        # =========================
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        # =========================
        # 2. CLOSE (preencher buracos)
        # =========================
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # =========================
        # 3. FILTRO POR ÁREA
        # =========================
        mask = self._filter_small_components(mask)

        # =========================
        # 4. GRABCUT (opcional)
        # =========================
        if self.use_grabcut:
            mask = self._apply_grabcut(image, mask)

        return mask

    def _filter_small_components(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        refined = np.zeros_like(mask)

        for i in range(1, num_labels):  # 0 = background
            area = stats[i, cv2.CC_STAT_AREA]

            if area >= self.min_area:
                refined[labels == i] = 255

        return refined

    def _apply_grabcut(self, image, mask):
        # bounding box do maior componente
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels <= 1:
            return mask

        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x = stats[largest_idx, cv2.CC_STAT_LEFT]
        y = stats[largest_idx, cv2.CC_STAT_TOP]
        w = stats[largest_idx, cv2.CC_STAT_WIDTH]
        h = stats[largest_idx, cv2.CC_STAT_HEIGHT]

        rect = (x, y, w, h)

        grabcut_mask = np.where(mask > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype("uint8")

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            image,
            grabcut_mask,
            rect,
            bgd_model,
            fgd_model,
            self.grabcut_iter,
            cv2.GC_INIT_WITH_MASK,
        )

        result = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype("uint8")

        return result