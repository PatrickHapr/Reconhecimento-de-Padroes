import os
import cv2
import numpy as np

def processar_multiescala(gray, scales=[1.0, 0.8, 1.2]):
    detections = []
    for scale in scales:
        scaled = cv2.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale))) if scale != 1.0 else gray.copy()

        thresh = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)

        kernel_size = max(2, int(3 * scale))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if 80 * (scale ** 2) < area < 2000 * (scale ** 2):
                if scale != 1.0:
                    contour = (contour / scale).astype(np.int32)
                detections.append((contour, area / (scale ** 2), scale))
    return detections

def filtrar_deteccoes_sobrepostas(detections, threshold=0.3):
    if len(detections) == 0:
        return []

    boxes = []
    for contour, area, scale in detections:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x+w, y+h, area, contour, scale])

    boxes = np.array(boxes, dtype=object)
    indices = np.argsort([box[4] for box in boxes])[::-1]
    
    keep = []
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        current_box = boxes[current][:4].astype(float)
        remaining_boxes = boxes[indices[1:]]

        new_indices = []
        for i, idx in enumerate(indices[1:]):
            other_box = remaining_boxes[i][:4].astype(float)

            x1 = max(current_box[0], other_box[0])
            y1 = max(current_box[1], other_box[1])
            x2 = min(current_box[2], other_box[2])
            y2 = min(current_box[3], other_box[3])

            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            union_area = (
                (current_box[2] - current_box[0]) * (current_box[3] - current_box[1]) +
                (other_box[2] - other_box[0]) * (other_box[3] - other_box[1]) -
                inter_area
            )

            iou = inter_area / union_area if union_area > 0 else 0
            if iou < threshold:
                new_indices.append(idx)

        indices = new_indices

    return [boxes[i] for i in keep]

def classificar_feijoes(img_original, gray, detections_filtradas):
    good_beans = 0
    bad_beans = 0
    img_resultado = img_original.copy()

    for det in detections_filtradas:
        x1, y1, x2, y2, area, contour, scale = det
        contour = np.array(contour, dtype=np.int32)

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        mean_color = cv2.mean(gray, mask=mask)[0]
        region_pixels = gray[mask == 255]
        std_color = np.std(region_pixels)

        is_dark = mean_color < 100
        is_uniform = std_color < 25

        if is_dark and is_uniform:
            good_beans += 1
            color = (0, 255, 0)
            label = "BOM"
        else:
            bad_beans += 1
            color = (0, 0, 255)
            label = "RUIM"

        cv2.drawContours(img_resultado, [contour], -1, color, 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(img_resultado, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    total = good_beans + bad_beans
    print("\n=== RESULTADOS MULTIESCALA ===")
    print(f"Feijões bons: {good_beans}")
    print(f"Feijões ruins: {bad_beans}")
    print(f"Total: {total}")
    if total > 0:
        print(f"% Bons: {good_beans / total * 100:.1f}%")
        print(f"% Ruins: {bad_beans / total * 100:.1f}%")

    return img_resultado

# === EXECUÇÃO ===
if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "feijoes.jpg")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem '{image_path}' não encontrada.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    detections = processar_multiescala(gray)
    filtradas = filtrar_deteccoes_sobrepostas(detections)
    resultado = classificar_feijoes(img, gray, filtradas)

    cv2.imshow("Feijões Detectados e Classificados", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "feijoes_classificados_multiescala.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, resultado)
