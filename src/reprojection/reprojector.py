import cv2
import numpy as np


class Reprojector:
    """
    Reprojeta máscaras de vistas perspectivas para o espaço equiretangular
    via inverse mapping vetorizado + cv2.remap.

    Convenção de eixos (alinhada com py360convert.e2p, usado na extração):
    - Equiretangular: coluna -> longitude em [-π, π); linha -> latitude em (π/2, -π/2).
    - Câmera default olha para +Z, com +X à direita e +Y para cima.
    - yaw (graus): rotação em torno de Y (positivo gira a câmera para a direita).
    - pitch (graus): rotação em torno de X (positivo aponta a câmera para cima;
      por isso pitch=-85 olha quase direto para o nadir).
    """

    def __init__(self, equi_shape):
        """
        equi_shape: tupla (H, W) da imagem 360° equiretangular original.
        """
        self.H, self.W = equi_shape
        # Pré-computa o vetor unitário 3D associado a cada pixel do equiretangular.
        # Reusado a cada chamada de reproject_mask.
        self._world_xyz = self._build_world_unit_vectors()

    def _build_world_unit_vectors(self):
        """
        Para cada pixel do equiretangular, calcula o vetor unitário 3D na esfera.
        Retorna array (H, W, 3) float32.
        """
        H, W = self.H, self.W

        # Centro do pixel (offset 0.5) reduz viés de amostragem.
        col = (np.arange(W, dtype=np.float32) + 0.5) / W
        row = (np.arange(H, dtype=np.float32) + 0.5) / H

        lon = col * 2.0 * np.pi - np.pi
        lat = np.pi / 2.0 - row * np.pi

        cos_lat = np.cos(lat)[:, None]
        sin_lat = np.sin(lat)[:, None]
        cos_lon = np.cos(lon)[None, :]
        sin_lon = np.sin(lon)[None, :]

        x = cos_lat * sin_lon
        y = np.broadcast_to(sin_lat, (H, W))
        z = cos_lat * cos_lon

        return np.stack([x, y, z], axis=-1).astype(np.float32)

    @staticmethod
    def _camera_to_world_rotation(yaw_deg, pitch_deg):
        """
        Matriz R que leva vetor do referencial câmera para o referencial mundo:
            v_world = R @ v_camera
        A inversa (R.T) leva mundo -> câmera.

        Composição: pitch primeiro (em torno de X), depois yaw (em torno de Y).
        """
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)

        Ry = np.array([
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ], dtype=np.float32)

        # Sinal de sp invertido para que pitch positivo aponte a câmera para cima.
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cp, sp],
            [0.0, -sp, cp],
        ], dtype=np.float32)

        return Ry @ Rx

    def reproject_mask(self, mask, metadata):
        """
        Reprojeta uma máscara binária da vista plana para o equiretangular.

        Parâmetros:
        - mask: np.ndarray (h, w) uint8, binária 0/255.
        - metadata: dict com 'yaw', 'pitch', 'fov' (graus).

        Retorna: np.ndarray (H, W) uint8 com a máscara reprojetada.
        """
        h, w = mask.shape[:2]
        fov_rad = np.deg2rad(metadata["fov"])

        # Distância focal em pixels (modelo pinhole, FOV horizontal).
        focal = (w / 2.0) / np.tan(fov_rad / 2.0)

        # Mundo -> câmera. arr @ R aplica R.T a cada vetor-linha (vetorizado).
        R = self._camera_to_world_rotation(metadata["yaw"], metadata["pitch"])
        xyz_cam = self._world_xyz @ R

        x_cam = xyz_cam[..., 0]
        y_cam = xyz_cam[..., 1]
        z_cam = xyz_cam[..., 2]

        # Pixels com z<=0 estão atrás da câmera, não são vistos.
        eps = 1e-6
        in_front = z_cam > eps

        # Projeção perspectiva. np.where evita divisão por zero em z=0.
        safe_z = np.where(in_front, z_cam, 1.0)
        x_plane = focal * x_cam / safe_z
        y_plane = focal * y_cam / safe_z

        # Plano -> coordenadas de pixel da imagem (linha cresce para baixo).
        map_x = (w / 2.0) + x_plane
        map_y = (h / 2.0) - y_plane

        in_bounds = (
            in_front
            & (map_x >= 0) & (map_x < w)
            & (map_y >= 0) & (map_y < h)
        )

        # Pixels inválidos viram -1 e caem no BORDER_CONSTANT=0 do remap.
        map_x = np.where(in_bounds, map_x, -1).astype(np.float32)
        map_y = np.where(in_bounds, map_y, -1).astype(np.float32)

        # INTER_NEAREST preserva binarização da máscara.
        return cv2.remap(
            mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    @staticmethod
    def fuse_masks(canvases, threshold=1):
        """
        Funde várias máscaras reprojetadas no equiretangular por voto.

        Para cada pixel, conta em quantas vistas ele apareceu. Pixels com pelo
        menos `threshold` votos passam.

        Com threshold=1 vira união (np.maximum). Com threshold>=2, exige overlap
        entre vistas e fica mais robusto a falsos positivos isolados, mas pode
        perder objeto quando o overlap é insuficiente (caso típico do nadir
        com 3 yaws espaçados 120°).
        """
        if not canvases:
            raise ValueError("Lista de canvases vazia.")

        stack = np.stack([(c > 127).astype(np.uint8) for c in canvases], axis=0)
        votes = stack.sum(axis=0)
        return ((votes >= threshold).astype(np.uint8)) * 255
