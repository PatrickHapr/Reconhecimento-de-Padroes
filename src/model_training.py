import cv2
import numpy as np

def extrair_caracteristicas(img_gray, contornos):
    """
    Extrai características dos contornos detectados de feijões.
    Retorna uma lista de vetores de características.
    """
    caracteristicas = []

    for contorno in contornos:
        mask = np.zeros(img_gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contorno], -1, 255, -1)

        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)
        circularidade = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0

        media_cor = cv2.mean(img_gray, mask=mask)[0]
        std_cor = np.std(img_gray[mask == 255])

        x, y, w, h = cv2.boundingRect(contorno)
        razao_largura_altura = w / h if h != 0 else 0

        classe = -1  # Placeholder, será definido posteriormente

        vetor = [area, perimetro, circularidade, media_cor, std_cor, razao_largura_altura, classe]
        caracteristicas.append(vetor)

    return caracteristicas
