import sys
import numpy as np
import cv2
import math

INPUT_IMAGES =  ['trabalho4/60.bmp', 'trabalho4/82.bmp', 'trabalho4/114.bmp', 'trabalho4/150.bmp', 'trabalho4/205.bmp']

BLOCK_SIZE = 501
C = -45
MORPHOLOGIC_KERNEL_SIZE = 5

BINARIZATION_THRESHOLD = 128

ALTURA_MIN = 5
LARGURA_MIN = 5
N_PIXELS_MIN = 50

MULTIPLE_RICE_THRESHOLD_FACTOR = 1.5
ADDITIONAL_RICE_FACTOR = 1.25


def binarize (img, threshold):
    img = np.where(img < threshold, 0, 1)
    img = img.astype(np.float32)

    return img


def label(img, largura_min, altura_min, n_pixels_min):
    img = cv2.convertScaleAbs(img)
    num_labels, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4, ltype=cv2.CV_32S)

    labeled_components = []
    for label in range(1, num_labels):
        left, top, width, height, area = stats[label]

        if width >= largura_min and height >= altura_min and area >= n_pixels_min:
            labeled_components.append({
                'label': label,
                'n_pixels': area,
                'T': top,
                'L': left,
                'B': top + height,
                'R': left + width
            })

    return labeled_components



def main ():
    for image in INPUT_IMAGES:
        img = cv2.imread(image)
        if img is None:
            print('Erro abrindo a imagem.\n')
            sys.exit()

        # cv2.imshow('Imagem original', img)

        # Binarização
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=BLOCK_SIZE, C=C)
        # cv2.imshow('Imagem binarizada', binary_img)

        # Abertura
        opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, np.ones((MORPHOLOGIC_KERNEL_SIZE, MORPHOLOGIC_KERNEL_SIZE), np.uint8))
        # cv2.imshow('Imagem aberta', opened_img)

        # Rotulagem
        opened_img = binarize(opened_img, BINARIZATION_THRESHOLD)
        components = label(opened_img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
        print('Componentes encontrados: %d' % len(components))

        for component in components:
            cv2.rectangle(img, (component['L'], component['T']), (component['R'], component['B']), (0, 255, 0), 1)

        cv2.imshow('Imagem com componentes', img)

        # Encontra a mediana do número de pixels brancos (arroz) em cada componente
        pixels = [component['n_pixels'] for component in components]
        pixels = np.array(pixels)
        median = np.median(pixels)

        # Encontra os componentes que provavelmente possuem mais de um grão de arroz e soma a quantidade de pixels deles.
        total_factor = 0
        for component in components:
            # Obs: São usados fatores de correção pois a mediana pode ser levemente afetada por levar em consideração
            # componentes com mais de um grão de arroz como um só componente.
            if component['n_pixels'] > MULTIPLE_RICE_THRESHOLD_FACTOR * median:
                total_factor += component['n_pixels']

        # Calcula a quantidade de arroz adicionais baseado na quantidade mediana de pixels de um arroz e
        # na quantidade total de pixels de arroz dos componentes que provavelmente possuem mais de um grão de arroz.
        calculate_additional_rices = math.floor(total_factor/(median * ADDITIONAL_RICE_FACTOR))
        print('Componentes adicionais: %d' % calculate_additional_rices)
        print('Número de grãos de arroz: %d' % (len(components) + calculate_additional_rices))
        print("\n--------------------\n")

        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main ()
