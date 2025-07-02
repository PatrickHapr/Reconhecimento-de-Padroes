import os
import cv2
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class FeijaoMLPClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        self.feijoes_data = {
            'caracteristicas': [],
            'labels': [],
            'imagens': []  # Para rastrear de qual imagem veio cada feijão
        }
        self.stats_finais = {}  # Para armazenar estatísticas finais

    def listar_imagens(self, pasta):
        """Lista todas as imagens na pasta"""
        formatos_validos = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        imagens = []
        
        for arquivo in os.listdir(pasta):
            if arquivo.lower().endswith(formatos_validos):
                imagens.append(os.path.join(pasta, arquivo))
        
        print(f"Encontradas {len(imagens)} imagens na pasta {pasta}")
        return imagens

    def detectar_feijoes(self, img_path):
        """Detecta feijões na imagem"""
        print(f"\nProcessando imagem: {os.path.basename(img_path)}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Erro ao carregar imagem: {img_path}")
            return None, None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        feijoes_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 10000:
                feijoes_contours.append(contour)
        
        print(f"Feijões detectados: {len(feijoes_contours)}")
        return img, gray, feijoes_contours

    def extrair_caracteristicas_simples(self, gray, contour):
        """Extrai características simples de cada feijão"""
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        pixels_feijao = gray[mask == 255]
        
        cor_media = np.mean(pixels_feijao)
        cor_desvio = np.std(pixels_feijao)
        cor_minima = np.min(pixels_feijao)
        cor_maxima = np.max(pixels_feijao)
        
        area = cv2.contourArea(contour)
        perimetro = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        proporcao = float(w) / h
        preenchimento = float(area) / (w * h)
        circularidade = 4 * np.pi * area / (perimetro * perimetro) if perimetro > 0 else 0
        
        return [
            cor_media, cor_desvio, cor_minima, cor_maxima,
            area, perimetro, proporcao, preenchimento, circularidade
        ]

    def simular_labels(self, caracteristicas_lista, img_nome):
        """Simula rótulos baseado nas características"""
        labels = []
        
        for caracteristicas in caracteristicas_lista:
            cor_media = caracteristicas[0]
            cor_desvio = caracteristicas[1]
            circularidade = caracteristicas[8]
            
            pontos_bom = 0
            
            # Ajuste nas regras para variar entre imagens
            if "ruim" in img_nome.lower():
                # Se o nome da imagem tem "ruim", tendência para classificar como ruim
                limiar_cor = 100
            else:
                limiar_cor = 80
                
            if cor_media < limiar_cor:
                pontos_bom += 1
            if cor_desvio < 20:
                pontos_bom += 1
            if circularidade > 0.5:
                pontos_bom += 1
            
            if pontos_bom >= 2:
                labels.append(1)  # BOM
            else:
                labels.append(0)  # RUIM
        
        return labels

    def processar_imagem(self, img_path):
        """Processa uma única imagem e retorna características e labels"""
        img, gray, contours = self.detectar_feijoes(img_path)
        if contours is None:
            return None, None
        
        caracteristicas_lista = []
        for contour in contours:
            caracteristicas = self.extrair_caracteristicas_simples(gray, contour)
            caracteristicas_lista.append(caracteristicas)
        
        img_nome = os.path.basename(img_path)
        labels = self.simular_labels(caracteristicas_lista, img_nome)
        
        return caracteristicas_lista, labels

    def processar_pasta_imagens(self, pasta):
        """Processa todas as imagens na pasta"""
        imagens = self.listar_imagens(pasta)
        if not imagens:
            print("Nenhuma imagem encontrada!")
            return False
        
        self.stats_finais['total_imagens'] = len(imagens)
        
        for img_path in imagens:
            caracteristicas, labels = self.processar_imagem(img_path)
            if caracteristicas:
                self.feijoes_data['caracteristicas'].extend(caracteristicas)
                self.feijoes_data['labels'].extend(labels)
                self.feijoes_data['imagens'].extend([os.path.basename(img_path)] * len(labels))
        
        print(f"\nTotal de feijões processados: {len(self.feijoes_data['labels'])}")
        print(f"Feijões BOM: {sum(self.feijoes_data['labels'])}")
        print(f"Feijões RUIM: {len(self.feijoes_data['labels']) - sum(self.feijoes_data['labels'])}")
        return True

    def treinar_rede_neural(self):
        """Treina a rede neural MLP com todos os dados"""
        if not self.feijoes_data['caracteristicas']:
            print("ERRO: Nenhum dado para treinar!")
            return None, None, None, None
        
        print("\n=== TREINANDO REDE NEURAL MLP ===")
        X = np.array(self.feijoes_data['caracteristicas'])
        y = np.array(self.feijoes_data['labels'])
        
        print(f"Formato dos dados: {X.shape}")
        print(f"Proporção BOM/RUIM: {sum(y)/len(y):.2f}/{1-sum(y)/len(y):.2f}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Treinando...")
        self.mlp.fit(X_train_scaled, y_train)
        
        y_pred = self.mlp.predict(X_test_scaled)
        accuracy = np.mean(y_pred == y_test)
        
        print("\n=== RESULTADOS ===")
        print(f"Acurácia: {accuracy:.3f}")
        print("\nRelatório de Classificação:")
        report = classification_report(y_test, y_pred, target_names=['RUIM', 'BOM'])
        print(report)
        
        print("\nMatriz de Confusão:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Armazenar estatísticas para o relatório final
        self.stats_finais.update({
            'total_feijoes': len(y),
            'feijoes_bom': sum(y),
            'feijoes_ruim': len(y) - sum(y),
            'formato_dados': X.shape,
            'acuracia': accuracy,
            'matriz_confusao': cm,
            'treino_size': len(X_train),
            'teste_size': len(X_test),
            'classification_report': report
        })
        
        return accuracy, X_test, y_test, y_pred

    def salvar_dataset(self, filename="dataset_feijoes.csv"):
        """Salva o dataset completo"""
        if not self.feijoes_data['caracteristicas']:
            print("ERRO: Nenhum dado para salvar!")
            return None
        
        nomes_colunas = [
            'cor_media', 'cor_desvio', 'cor_minima', 'cor_maxima',
            'area', 'perimetro', 'proporcao', 'preenchimento', 'circularidade',
            'classe', 'imagem_origem'
        ]
        
        dados = []
        for i, carac in enumerate(self.feijoes_data['caracteristicas']):
            linha = carac + [self.feijoes_data['labels'][i]] + [self.feijoes_data['imagens'][i]]
            dados.append(linha)
        
        df = pd.DataFrame(dados, columns=nomes_colunas)
        df.to_csv(filename, index=False)
        
        print(f"\nDataset salvo em: {filename}")
        print(f"Total de amostras: {len(df)}")
        print("\nPrimeiras 5 linhas:")
        print(df.head())
        
        self.stats_finais['dataset_filename'] = filename
        self.stats_finais['dataset_shape'] = df.shape
        
        return df

    def visualizar_resultados(self, img_path):
        """Mostra os resultados para uma imagem específica"""
        img, gray, contours = self.detectar_feijoes(img_path)
        if contours is None:
            return None
        
        caracteristicas_lista = []
        for contour in contours:
            caracteristicas = self.extrair_caracteristicas_simples(gray, contour)
            caracteristicas_lista.append(caracteristicas)
        
        X = np.array(caracteristicas_lista)
        X_scaled = self.scaler.transform(X)
        predicoes = self.mlp.predict(X_scaled)
        probabilidades = self.mlp.predict_proba(X_scaled)
        
        img_resultado = img.copy()
        bons = 0
        ruins = 0
        
        for i, contour in enumerate(contours):
            if predicoes[i] == 1:
                cor = (0, 255, 0)  # Verde
                texto = f"BOM ({probabilidades[i][1]:.2f})"
                bons += 1
            else:
                cor = (0, 0, 255)  # Vermelho
                texto = f"RUIM ({probabilidades[i][0]:.2f})"
                ruins += 1
            
            cv2.drawContours(img_resultado, [contour], -1, cor, 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(img_resultado, texto, (x, y-5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, cor, 1)
        
        total = bons + ruins
        print(f"\n=== RESULTADOS PARA {os.path.basename(img_path)} ===")
        print(f"Feijões BOM: {bons}")
        print(f"Feijões RUIM: {ruins}")
        print(f"Proporção: {bons/total:.1%} BOM / {ruins/total:.1%} RUIM")
        
        cv2.imshow(f"Classificação: {os.path.basename(img_path)}", img_resultado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        resultado_path = f"resultado_{os.path.basename(img_path)}"
        cv2.imwrite(resultado_path, img_resultado)
        print(f"Resultado salvo em: {resultado_path}")
        
        return img_resultado

    def exibir_relatorio_final(self):
        """Exibe um relatório final completo com todas as estatísticas"""
        print("\n" + "="*60)
        print("                 RELATÓRIO FINAL COMPLETO")
        print("="*60)
        
        print(f"\n📊 ESTATÍSTICAS GERAIS:")
        print(f"   • Total de imagens processadas: {self.stats_finais.get('total_imagens', 0)}")
        print(f"   • Total de feijões detectados: {self.stats_finais.get('total_feijoes', 0)}")
        print(f"   • Feijões classificados como BOM: {self.stats_finais.get('feijoes_bom', 0)}")
        print(f"   • Feijões classificados como RUIM: {self.stats_finais.get('feijoes_ruim', 0)}")
        
        if self.stats_finais.get('total_feijoes', 0) > 0:
            prop_bom = self.stats_finais['feijoes_bom'] / self.stats_finais['total_feijoes']
            prop_ruim = self.stats_finais['feijoes_ruim'] / self.stats_finais['total_feijoes']
            print(f"   • Proporção BOM/RUIM: {prop_bom:.2f}/{prop_ruim:.2f}")
        
        print(f"\n🧠 INFORMAÇÕES DA REDE NEURAL:")
        print(f"   • Formato dos dados: {self.stats_finais.get('formato_dados', 'N/A')}")
        print(f"   • Tamanho do conjunto de treino: {self.stats_finais.get('treino_size', 0)}")
        print(f"   • Tamanho do conjunto de teste: {self.stats_finais.get('teste_size', 0)}")
        print(f"   • Arquitetura: MLP com camadas ocultas (100, 50)")
        print(f"   • Função de ativação: ReLU")
        print(f"   • Otimizador: Adam")
        
        print(f"\n🎯 DESEMPENHO DO MODELO:")
        print(f"   • Acurácia alcançada: {self.stats_finais.get('acuracia', 0):.3f}")
        
        if 'matriz_confusao' in self.stats_finais:
            cm = self.stats_finais['matriz_confusao']
            print(f"\n📈 MATRIZ DE CONFUSÃO:")
            print(f"                 Predito")
            print(f"              RUIM    BOM")
            print(f"   Real RUIM  {cm[0][0]:4d}   {cm[0][1]:4d}")
            print(f"        BOM   {cm[1][0]:4d}   {cm[1][1]:4d}")
            
            # Calcular métricas detalhadas
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            precision_ruim = tn / (tn + fn) if (tn + fn) > 0 else 0
            precision_bom = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_ruim = tn / (tn + fp) if (tn + fp) > 0 else 0
            recall_bom = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            print(f"\n📊 MÉTRICAS DETALHADAS:")
            print(f"   • Precisão RUIM: {precision_ruim:.3f}")
            print(f"   • Precisão BOM:  {precision_bom:.3f}")
            print(f"   • Recall RUIM:   {recall_ruim:.3f}")
            print(f"   • Recall BOM:    {recall_bom:.3f}")
        
        print(f"\n💾 ARQUIVOS GERADOS:")
        if 'dataset_filename' in self.stats_finais:
            print(f"   • Dataset: {self.stats_finais['dataset_filename']}")
            print(f"   • Dimensões do dataset: {self.stats_finais['dataset_shape']}")
        print(f"   • Imagens de resultado salvas com prefixo 'resultado_'")
        
        print(f"\n🔧 CARACTERÍSTICAS EXTRAÍDAS (9 features):")
        features = [
            "1. Cor média", "2. Desvio padrão da cor", "3. Cor mínima", 
            "4. Cor máxima", "5. Área", "6. Perímetro", 
            "7. Proporção (largura/altura)", "8. Preenchimento", "9. Circularidade"
        ]
        for feature in features:
            print(f"   • {feature}")
        
        print("\n" + "="*60)
        print("           PROCESSAMENTO FINALIZADO COM SUCESSO!")
        print("="*60)

# === USO DO SISTEMA ===
if __name__ == "__main__":
    classificador = FeijaoMLPClassifier()
    pasta_imagens = os.path.join("data", "raw")
    
    try:
        # 1. Processar todas as imagens na pasta
        print("=== PASSO 1: PROCESSANDO IMAGENS ===")
        if not classificador.processar_pasta_imagens(pasta_imagens):
            exit()
        
        # 2. Salvar dataset completo
        print("\n=== PASSO 2: SALVANDO DATASET ===")
        df = classificador.salvar_dataset()
        
        # 3. Treinar rede neural com todos os dados
        print("\n=== PASSO 3: TREINANDO REDE NEURAL ===")
        classificador.treinar_rede_neural()
        
        # 4. Visualizar resultados para cada imagem
        print("\n=== PASSO 4: VISUALIZANDO RESULTADOS ===")
        imagens = classificador.listar_imagens(pasta_imagens)
        for img_path in imagens:
            classificador.visualizar_resultados(img_path)
        
        # 5. NOVO: Exibir relatório final completo
        classificador.exibir_relatorio_final()
        
    except Exception as e:
        print(f"ERRO: {e}")