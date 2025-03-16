import cv2
import pandas as pd
import datetime
from ultralytics import YOLO  # Importando o modelo YOLO
import time  # Para controlar o intervalo de detecção
import os  # Para manipulação de arquivos
import sys

# Perguntar qual arquivo .mp4 será processado
video_path = input("Digite o caminho do arquivo .mp4 que deseja processar: ").strip()

# Verificar se o arquivo existe
if not os.path.exists(video_path):
    print(f"Erro: O arquivo {video_path} não foi encontrado.")
    sys.exit()

# Carregar o modelo YOLO
modelo = YOLO('yolov10s.pt')  # Ajuste o nome do modelo conforme necessário

# Abrir o vídeo original para processamento
cap = cv2.VideoCapture(video_path)

# Verificar se o vídeo foi aberto corretamente
if not cap.isOpened():
    print("Erro ao abrir o arquivo de vídeo.")
    sys.exit()

# Obter as propriedades do vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Definir o codec e criar o objeto de escrita do vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Formatar a data e hora no formato 'YYYYMMDD_HHMMSS'
video_out = cv2.VideoWriter(f'captura_video_{timestamp}.mp4', fourcc, frame_rate, (frame_width, frame_height))

# Variável para controlar o tempo de detecção
last_detection_time = time.time()  # Tempo da última detecção
interval = 1  # Intervalo entre as detecções (1 segundo)

# Criar uma lista para armazenar os dados das detecções
detecoes = []

# Processamento de vídeo com YOLO
while True:
    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    # Verificar se já se passou 1 segundo desde a última detecção
    if current_time - last_detection_time >= interval:
        # Realizar a previsão no quadro capturado com YOLO
        results = modelo.predict(source=frame)

        # A renderização dos resultados modifica a imagem diretamente
        img = results[0].plot()  # Usando 'plot()' para renderizar as detecções no quadro

        # Adicionar detecções à lista
        for result in results[0].boxes:
            # Para cada caixa de detecção, capturamos os dados
            description = result.cls[0]  # Obter o código da classe
            confidence = result.conf[0]  # Obter a confiança
            xmin, ymin, xmax, ymax = result.xyxy[0].tolist()  # Coordenadas do bounding box

            # Converter o código da classe para a descrição (se disponível no modelo)
            class_names = modelo.names  # Obter os nomes das classes do modelo YOLO
            class_description = class_names[int(description)]  # Converter para o nome da classe

            # Registrar os dados na lista de detecções, incluindo data e hora
            data_hora_atual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Formatar a data e hora no formato 'YYYY-MM-DD HH:MM:SS'
            detecoes.append([class_description, confidence, data_hora_atual, xmin, ymin, xmax, ymax])

        # Atualizar o tempo da última detecção
        last_detection_time = current_time
    else:
        # Caso não tenha detecção no intervalo, usar o quadro original
        img = frame

    # Salvar o vídeo processado
    video_out.write(img)

    # Exibir o vídeo processado
    cv2.imshow('Vídeo Processado', img)

    # Se a tecla ESC (código 27) for pressionada, sai do loop
    if cv2.waitKey(1) == 27:  # 27 é o código da tecla ESC
        break

# Libera a captura de vídeo e fecha o arquivo de vídeo processado
cap.release()
video_out.release()
cv2.destroyAllWindows()

# Criar o DataFrame e salvar em Excel
df = pd.DataFrame(detecoes, columns=["Classe", "Confiança", "Data e Hora", "Xmin", "Ymin", "Xmax", "Ymax"])
df.to_excel(f'detecoes_{timestamp}.xlsx', index=False)

print(f"Arquivo de Excel gerado: detecoes_{timestamp}.xlsx")

# Encerrar o código após gerar o arquivo Excel
sys.exit()
