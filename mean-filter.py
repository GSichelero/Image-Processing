"""
Objetivo: implemente 3 algoritmos para o filtro da média:
- Algoritmo “ingênuo”.
- Filtro separável (com ou sem aproveitar as somas anteriores).
- Algoritmo com imagens integrais.


Notas:
- Coloque as 3 implementações no mesmo arquivo, junto com um programa principal que permita testá-las.
- Para imagens coloridas, processar cada canal RGB independentemente.
- Tratamento das margens: na implementação com imagens integrais, fazer média com janelas menores; nas outras pode simplesmente ignorar posições cujas janelas ficariam fora da imagem.
- O pacote tem algumas imagens para comparação. Se estiver usando OpenCV, compare os resultados com os da função blur da biblioteca (exceto pelas margens, o resultado deve ser igual!).
"""

import sys
import timeit
import numpy as np
import cv2

INPUT_IMAGE =  'pacote_trabalho2/Exemplos/b01 - Original.bmp'
WINDOW = 5

def naive_mean_filter(img, window_size):
    height = img.shape[0]
    width = img.shape[1]
    color_channels = img.shape[2]
    half_window = window_size // 2
    output_img = np.zeros((height, width, color_channels), np.uint8)
    for h in range(color_channels):
        for i in range(height):
            for j in range(width):
                sum_values = 0
                count_pixels = 0
                for k in range(-half_window, half_window + 1):
                    for l in range(-half_window, half_window + 1):
                        if i+k >= 0 and i+k < height and j+l >= 0 and j+l < width:
                            sum_values += img[i+k, j+l, h]
                            count_pixels += 1
                output_img[i, j, h] = sum_values // count_pixels

    return output_img



def main ():
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    cv2.imshow('Imagem original', img)

    start_time = timeit.default_timer()
    output_img = naive_mean_filter(img, WINDOW)
    print('Tempo Algoritmo “ingênuo”: %f' % (timeit.default_timer() - start_time))
    cv2.imshow('Imagem filtrada Algoritmo “ingênuo”', output_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main ()
