#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'arroz.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 8
LARGURA_MIN = 8
N_PIXELS_MIN = 64

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!

    img = np.where(img < threshold, 0, 1)
    img = img.astype(np.float32)

    return img

#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.

    def inunda(img, x, y, label, max_width, max_height, component):
        if x < 0 or x >= max_width or y < 0 or y >= max_height:
            return
        if img[y,x] != 1:
            return
        img[y,x] = label
        component['n_pixels'] += 1
        component['T'] = min(component['T'], y)
        component['L'] = min(component['L'], x)
        component['B'] = max(component['B'], y)
        component['R'] = max(component['R'], x)
        inunda(img, x+1, y, label, max_width, max_height, component)
        inunda(img, x-1, y, label, max_width, max_height, component)
        inunda(img, x, y+1, label, max_width, max_height, component)
        inunda(img, x, y-1, label, max_width, max_height, component)
        
    
    max_height = img.shape[0]
    max_width = img.shape[1]

    components = []
    label = 2
    for y in range(max_height):
        for x in range(max_width):
            if img[y,x] == 1:
                component = {
                    'label': label,
                    'n_pixels': 0,
                    'T': max_height,
                    'L': max_width,
                    'B': 0,
                    'R': 0
                }

                inunda(img, x, y, label, max_width, max_height, component)

                if (component['n_pixels'] >= n_pixels_min and
                (component['R'] - component['L']) >= largura_min and
                (component['B'] - component['T']) >= altura_min):
                    components.append(component)
                
                label += 1

    return components





#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
