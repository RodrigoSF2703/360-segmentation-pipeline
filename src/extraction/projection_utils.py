import py360convert
import numpy as np


def equirectangular_to_perspective(
    image: np.ndarray,
    yaw: float,
    pitch: float,
    fov: float = 90,
    output_size: tuple = (512, 512)
) -> np.ndarray:
    """
    Converte uma imagem equiretangular (360°) em uma vista perspectiva.

    Parâmetros:
    ----------
    image : np.ndarray
        Imagem de entrada no formato equiretangular (H x W x C)
    yaw : float
        Rotação horizontal (em graus)
    pitch : float
        Rotação vertical (em graus)
    fov : float
        Campo de visão da câmera virtual (em graus)
    output_size : tuple
        Tamanho da imagem de saída (altura, largura)

    Retorna:
    -------
    np.ndarray
        Imagem em perspectiva
    """

    perspective = py360convert.e2p(
        image,
        fov_deg=fov,
        u_deg=yaw,
        v_deg=pitch,
        out_hw=output_size
    )

    return perspective