import cv2
import pandas as pd
import datetime
from ultralytics import YOLO  # Importando o modelo YOLO
import time  # Para controlar o intervalo de detecção
import os  # Importar para verificar o caminho do MonaServer
import signal
import sys
import subprocess  # Para iniciar o MonaServer de maneira mais robusta

constante1 = "rtmp://"
constante2 = ":1935/"

# Pergunta para o usuário
entrada = input("Você deseja usar a webcam ou o RTMP? (Digite 'webcam' ou 'drone'): ").strip().lower()

# Condicional para configurar o fluxo de vídeo de acordo com a escolha do usuário
if entrada == "webcam":
    # Usar a webcam
    cap = cv2.VideoCapture(0)  # Captura da webcam
elif entrada == "drone":
    # Executar o MonaServer
    monaserver_path = r"C:\Users\henferraz\Desktop\Dev\Detecção e processamento com drone\MonaServer_Win32\MonaServer.exe"  # Corrigido com prefixo 'r' para caminho no Windows
    if os.path.exists(monaserver_path):
        monaserver_process = subprocess.Popen(monaserver_path)  # Usando subprocess para iniciar o MonaServer
    else:
        print("Aviso: MonaServer.exe não encontrado. Certifique-se de que ele está no caminho correto.")
        exit()  # Sai do código se MonaServer não for encontrado

    time.sleep(5)
    print("Continuando conexão...")

    # Conexão via RTMP
    ip_computador = input("Digite o IP do computador (o mesmo IP deve estar configurado no drone): ").strip()
    url_stream = constante1 + ip_computador + constante2
    cap = cv2.VideoCapture(url_stream) 
else:
    print("Opção inválida. Por favor, digite 'webcam' ou 'drone'.")
    exit()

# Verificar se a captura foi bem-sucedida
if not cap.isOpened():
    print("Erro ao abrir o fluxo de vídeo.")
    exit()

# Obter as propriedades do vídeo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Definir o codec e criar o objeto de escrita do vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Formatar a data e hora no formato 'YYYYMMDD_HHMMSS'
video_out = cv2.VideoWriter(f'captura_video_{timestamp}.mp4', fourcc, frame_rate, (frame_width, frame_height))

# Variável para controlar se o vídeo está sendo processado ou não
processando_video = False

# Criar uma lista para armazenar os dados das detecções
detecoes = []

# Grava o vídeo sem processamento
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Erro ao receber frame. Encerrando...")
        break
    
    # Salvar o vídeo sem processamento
    video_out.write(frame)  # Salvar o quadro original

    # Exibir o quadro original sem qualquer modificação (imagem sem processamento)
    cv2.imshow('Câmera', frame)
    
    # Se a tecla ESC (código 27) for pressionada, sai do loop
    if cv2.waitKey(1) == 27:  # 27 é o código da tecla ESC
        break

# Libera a captura de vídeo e salva o arquivo de vídeo original
cap.release()
video_out.release()

cv2.destroyAllWindows()

# Agora, processar o vídeo original com YOLO
modelo = YOLO('yolov10s.pt')  # Carregar o modelo YOLO (ajustar conforme o seu modelo)

# Abrir o vídeo original para processamento
cap = cv2.VideoCapture(f'captura_video_{timestamp}.mp4')

# Configurar o VideoWriter para salvar o vídeo processado
out_processado = cv2.VideoWriter(f'captura_video_{timestamp}_processado.mp4', fourcc, frame_rate, (frame_width, frame_height))

# Controle de tempo para realizar a detecção a cada 1 segundo
last_detection_time = time.time()  # Tempo da última detecção
interval = 1  # Intervalo entre as detecções (1 segundo)

# Detecção de objetos e registro em lista
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
    out_processado.write(img)
    
    # Se a tecla ESC (código 27) for pressionada, sai do loop
    if cv2.waitKey(1) == 27:  # 27 é o código da tecla ESC
        break

# Libera a captura de vídeo, fecha as janelas abertas e salva o arquivo de vídeo processado
cap.release()
out_processado.release()
cv2.destroyAllWindows()

# Criar o DataFrame e salvar em Excel
df = pd.DataFrame(detecoes, columns=["Classe", "Confiança", "Data e Hora", "Xmin", "Ymin", "Xmax", "Ymax"])
df.to_excel(f'detecoes_{timestamp}.xlsx', index=False)

print(f"Arquivo de Excel gerado: detecoes_{timestamp}.xlsx")

# Se o MonaServer foi iniciado (caso tenha sido necessário para o fluxo RTMP), matar o processo
if entrada == "drone" and 'monaserver_process' in locals():
    try:
        monaserver_process.terminate()  # Tentar encerrar o processo de forma elegante
        print("MonaServer encerrado com sucesso.")
    except Exception as e:
        print(f"Erro ao tentar encerrar o MonaServer: {e}")

# Encerrar o código após gerar o arquivo Excel
sys.exit()
