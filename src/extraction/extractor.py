import os
import cv2
import json
from typing import List, Dict
from .projection_utils import equirectangular_to_perspective


class ViewExtractor:
    """
    Classe responsável por extrair vistas perspectiva de imagens 360°.
    """

    def __init__(self, fov: float = 90, output_size: tuple = (512, 512)):
        self.fov = fov
        self.output_size = output_size

    def extract_single_view(
        self,
        image,
        yaw: float,
        pitch: float
    ):
        """
        Extrai uma única vista da imagem.
        """
        return equirectangular_to_perspective(
            image=image,
            yaw=yaw,
            pitch=pitch,
            fov=self.fov,
            output_size=self.output_size
        )

    def extract_multiple_views(
        self,
        image,
        yaw_list: List[float],
        pitch: float,
    ) -> List[Dict]:
        """
        Extrai múltiplas vistas variando o yaw.

        Retorna uma lista de dicionários contendo:
        - imagem
        - metadados (yaw, pitch)
        """
        views = []

        for yaw in yaw_list:
            view = self.extract_single_view(image, yaw, pitch)

            views.append({
                "image": view,
                "yaw": yaw,
                "pitch": pitch,
                "view_id": f"yaw{int(yaw)}_pitch{int(pitch)}"
            })

        return views

    def save_views(
        self,
        views,
        output_dir: str,
        base_name: str,
        source_image: str
    ):
        """
        Salva as vistas no disco com imagem + metadados JSON.
        """

        os.makedirs(output_dir, exist_ok=True)

        for v in views:
            view_id = f"yaw{int(v['yaw'])}_pitch{int(v['pitch'])}"

            # Caminho da imagem
            img_filename = f"{base_name}_{view_id}.png"
            img_path = os.path.join(output_dir, img_filename)

            # Salvar imagem
            cv2.imwrite(img_path, v["image"])

            # Criar metadados
            metadata = {
                "source_image": source_image,
                "view_id": view_id,
                "yaw": v["yaw"],
                "pitch": v["pitch"],
                "fov": self.fov,
                "output_size": list(self.output_size)
            }

            # Caminho do JSON
            json_filename = f"{base_name}_{view_id}.json"
            json_path = os.path.join(output_dir, json_filename)

            # Salvar JSON
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=4)


    def extract_grid_views(self, image, yaw_list, pitch_list):
        """
        Extrai vistas em formato de malha (grid), combinando yaw e pitch.
        """
        views = []

        for pitch in pitch_list:
            for yaw in yaw_list:
                view = self.extract_single_view(image, yaw, pitch)

                views.append({
                    "image": view,
                    "yaw": yaw,
                    "pitch": pitch,
                    "view_id": f"yaw{int(yaw)}_pitch{int(pitch)}"
                })

        return views