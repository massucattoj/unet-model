# Deep Learning - Unet Model
  > Python 3.0
  > Keras
  
  Uma modelo de rede neural convolucional para segmentações de imagens rápidas e precisas. Sendo bastante 
  utilizadas tratando-se de imagens medicinais. 
  
  Modelo composto de duas partes, a primeira um processo de downsampling e então um processo de upsampling 
  aplicando uma operação de concatenção entre as camadas obtidas.

## Model Architecture
<p align="center">
   <img src="u-net-architecture.png" />
</p>

  Exemplo de arquitetura com uma imagem de 32x32 pixels na resolução mais baixa. Aonde cada caixa azul 
  corresponde ao mapa de caracteristicas. O numero de canais é denotado no topo da caixa. O tamanho dos
  eixos x e y são mostrados no canto inferior esquerdo. Caixas brancas representam copias do mapa de
  caracteristicas e as setas os diferentes tipos de operação.

### Material
- More about Deep Learning Unet could be visualize in this paper: [https://arxiv.org/pdf/1505.04597.pdf]
