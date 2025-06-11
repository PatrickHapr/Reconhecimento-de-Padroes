import cv2
import numpy as np

def estrategia_hough_circles(image_path):
    """
    Estratégia usando detecção de círculos de Hough
    Feijões têm formato aproximadamente elíptico
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem '{image_path}' não encontrada.")
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pré-processamento específico para Hough
    # 1. Equalização de histograma
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 2. Suavização Gaussiana
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 3. Detecção de círculos/elipses com HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,           # Resolução do acumulador
        minDist=15,     # Distância mínima entre centros
        param1=50,      # Threshold superior para detecção de bordas
        param2=25,      # Threshold do acumulador
        minRadius=8,    # Raio mínimo
        maxRadius=25    # Raio máximo
    )
    
    good_beans = 0
    bad_beans = 0
    detected_circles = []
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        for (x, y, r) in circles:
            # Criar máscara circular
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Analisar a região
            mean_intensity = cv2.mean(gray, mask=mask)[0]
            
            # Verificar se a região contém um objeto real
            region_pixels = gray[mask == 255]
            if len(region_pixels) < 50:  # Muito poucos pixels
                continue
                
            std_intensity = np.std(region_pixels)
            
            # Classificação baseada em intensidade
            if mean_intensity < 110:
                good_beans += 1
                color = (0, 255, 0)
                thickness = 3
            else:
                bad_beans += 1
                color = (0, 0, 255)
                thickness = 2
            
            # Desenhar círculo
            cv2.circle(img, (x, y), r, color, thickness)
            cv2.putText(img, f"{int(mean_intensity)}", (x-10, y-r-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            detected_circles.append((x, y, r, mean_intensity))
    
    return img, good_beans, bad_beans, detected_circles

def estrategia_template_matching(image_path):
    """
    Estratégia usando template matching
    Cria templates de feijões e procura por eles
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Imagem '{image_path}' não encontrada.")
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Criar templates sintéticos de feijões
    templates = []
    
    # Template 1: Elipse pequena (feijão bom)
    template1 = np.zeros((30, 20), dtype=np.uint8)
    cv2.ellipse(template1, (10, 15), (8, 12), 0, 0, 360, 255, -1)
    templates.append(('bom_pequeno', template1))
    
    # Template 2: Elipse média
    template2 = np.zeros((40, 25), dtype=np.uint8)
    cv2.ellipse(template2, (12, 20), (10, 16), 0, 0, 360, 255, -1)
    templates.append(('bom_medio', template2))
    
    # Template 3: Formato irregular (feijão ruim)
    template3 = np.zeros((35, 22), dtype=np.uint8)
    cv2.ellipse(template3, (11, 17), (9, 14), 0, 0, 360, 255, -1)
    # Adicionar "rachadura"
    cv2.line(template3, (5, 10), (17, 24), 0, 2)
    templates.append(('ruim', template3))
    
    # Pré-processamento da imagem
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    detections = []
    good_beans = 0
    bad_beans = 0
    
    # Para cada template
    for template_name, template in templates:
        # Template matching
        result = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
        
        # Encontrar matches acima do threshold
        threshold = 0.6
        locations = np.where(result >= threshold)
        
        for pt in zip(*locations[::-1]):  # Trocar x e y
            x, y = pt
            w, h = template.shape[::-1]
            
            # Verificar sobreposição com detecções anteriores
            overlap = False
            for prev_x, prev_y, prev_w, prev_h, _ in detections:
                if (abs(x - prev_x) < 15 and abs(y - prev_y) < 15):
                    overlap = True
                    break
            
            if not overlap:
                # Analisar a região real
                roi = gray[y:y+h, x:x+w]
                if roi.size > 0:
                    mean_intensity = np.mean(roi)
                    
                    detections.append((x, y, w, h, template_name))
                    
                    if 'bom' in template_name:
                        good_beans += 1
                        color = (0, 255, 0)
                    else:
                        bad_beans += 1
                        color = (0, 0, 255)
                    
                    # Desenhar retângulo
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(img, template_name[:4], (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return img, good_beans, bad_beans, detections

# Executar ambas as estratégias
if __name__ == "__main__":
    print("=== TESTANDO ESTRATÉGIA HOUGH CIRCLES ===")
    try:
        result1, good1, bad1, circles = estrategia_hough_circles("feijoes.jpg")
        print(f"Hough - Bons: {good1}, Ruins: {bad1}, Total: {good1+bad1}")
        
        cv2.imshow("Hough Circles", cv2.resize(result1, (800, 600)))
        cv2.waitKey(0)
        
    except Exception as e:
        print(f"Erro Hough: {e}")
    
    print("\n=== TESTANDO ESTRATÉGIA TEMPLATE MATCHING ===")
    try:
        result2, good2, bad2, detections = estrategia_template_matching("feijoes.jpg")
        print(f"Template - Bons: {good2}, Ruins: {bad2}, Total: {good2+bad2}")
        
        cv2.imshow("Template Matching", cv2.resize(result2, (800, 600)))
        cv2.waitKey(0)
        
    except Exception as e:
        print(f"Erro Template: {e}")
    
    cv2.destroyAllWindows()