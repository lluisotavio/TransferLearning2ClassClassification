[![IMAGE ALT TEXT](http://img.youtube.com/vi/ZGaMlhvkYdw/0.jpg)](http://www.youtube.com/watch?v=ZGaMlhvkYdw "Video Title")

# TransferLearning2ClassClassification

Este projeto tem como objetivo automatizar o teste de diferentes redes convolucionais implementadas no keras para classificação de
2 classes com o uso de transferência de aprendizado (pesos das redes pré treinadas com ImageNet). Mais especificamente, chamamos 
aqui as duas classes como "armas" ou "não armas".

As rotinas foram criadas para rodarem de forma idependente.

## Treinamento

Para uma execução padrão, apenas crie e altere os caminhos das pastas utilizadas no Config.py, insira as imagens com as classes
para treinamento das duas classes nas pastas de treinamento (ARMAS_FOLDER e NAO_ARMAS_FOLDER) e rode o treinamento.py. Segue um
exemplo do Config.py abaixo:

ARMAS_FOLDER = 'E:\\datasets\\armas\\'
NAO_ARMAS_FOLDER = 'E:\\datasets\\all\\'
ARMAS_TESTE_FOLDER = 'E:\\datasets\\armas_teste\\'
ARMAS_TESTE_NAO_FOLDER = 'E:\\datasets\\all_teste\\'
RESULTADOS_FOLDER = 'E:\\datasets\\resultado\\'
MODELO_SAVE_FOLDER = 'E:\\datasets\\resultado\\modelos\\'

AMOSTRAS_POR_CLASSE = 0
EPOCAS = 30
BATCH_SIZE = 16
label_nome = {0:'nao_arma', 1:'arma'}
\# Disponíveis modelos "INCEPTIONV3" "VGG16" e "MobileNetV2"
MODELO = 'vgg16'

PATH_MODELO = 'E:\\datasets\\resultado\\modelos\\_vgg16weights.03-0.07_dropout.h5'
PATH_VIDEO_PARA_TESTE = 'C:\\Users\\LLUIS\\Documents\\Wondershare Filmora\\Output\\video_teste_filmes.mp4'
PATH_VIDEO_TESTE_SAVE_PATH = 'C:\\Users\\LLUIS\\Desktop\\filme_classificacao.mp4'

## Teste

Com o treinamento feito,  e utilizando os pesos gerados (salvos em MODELO_SAVE_FOLDER e definindo qual será usado em PATH_MODELO),
você poderá utilizar seu modelo para classificar suas imagens dentro das pastas de treinamento (ARMAS_TESTE_FOLDER, 
ARMAS_TESTE_NAO_FOLDER) rodando teste.py.

Também com os pesos gerados, poderá testar seu modelo com a webcam utilizando o teste_com_webcam.py e seus próprios vídeos
(PATH_VIDEO_PARA_TESTE para definir qual vídeo usar e PATH_VIDEO_TESTE_SAVE_PATH o vídeo com as marcas da classificação que
será salvo)

## Detecção

Há também uma versão para detecção de imagens (Colab_Treinamento_Teste_Yolov3_Luis_Rafael_Vinicius.ipynb), baseado na yolo-v3, apresentado em https://pjreddie.com/darknet/yolo/.
Esta versão é baseada no repositório https://github.com/AlexeyAB/darknet.

## Dataset

Um dataset está disponível em
https://drive.google.com/drive/folders/1IEe_yn8mnjYf9mWNCiDZ7hKIkag6cjsA?usp=sharing





Fique livre para utilização deste código como bem desejar,

Luís Otávio Mendes da Silva
Vinícius de Souza Rios

lluisotavio@gmail.com

## Resultados

Alguns resultados gerados pela aplicação deste código foram mostrados neste vídeo

https://www.youtube.com/watch?v=ZGaMlhvkYdw
