import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

INPUT_IMAGES =  ['trabalho4/60.bmp', 'trabalho4/82.bmp', 'trabalho4/114.bmp', 'trabalho4/150.bmp', 'trabalho4/205.bmp']

BLOCK_SIZE = 501
C = -45
MORPHOLOGIC_KERNEL_SIZE = 5

ALTURA_MIN = 1
LARGURA_MIN = 1
N_PIXELS_MIN = 1

ROUND_THRESHOLD = 0.9


def custom_round(number):
    integer_part = int(number)
    decimal_part = number - integer_part
    if decimal_part > ROUND_THRESHOLD:
        return math.ceil(number)
    else:
        return math.floor(number)


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
        opened_img = binarize(opened_img, 128)
        components = label(opened_img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
        print('Componentes encontrados: %d' % len(components))

        for component in components:
            cv2.rectangle(img, (component['L'], component['T']), (component['R'], component['B']), (0, 255, 0), 1)

        cv2.imshow('Imagem com componentes', img)

        pixels = [component['n_pixels'] for component in components]
        pixels = np.array(pixels)
        mean = np.mean(pixels)
        std = np.std(pixels)
        median = np.median(pixels)
        print('Média: %f' % mean)
        print('Desvio padrão: %f' % std)
        print('Mediana: %f' % median)

        total_factor = 0
        for component in components:
            if component['n_pixels'] > 1.5 * median:
                total_factor += component['n_pixels']
                print(f'Componente: {component["n_pixels"]} = {str(round(component["n_pixels"] / (median * 1), 2))}')

        calculate_additional_rices = math.ceil(total_factor/(median * 1.25))
        print(calculate_additional_rices)
        calculate_additional_rices = custom_round(total_factor/(median * 1.25))
        print(calculate_additional_rices)
        calculate_additional_rices = round(total_factor/(median * 1.25), 0)
        print(calculate_additional_rices)
        calculate_additional_rices = math.floor(total_factor/(median * 1.25))
        print(calculate_additional_rices)
        print('Componentes adicionais: %d' % calculate_additional_rices)
        print('Componentes totais: %d' % (len(components) + calculate_additional_rices))
        print("\n--------------------\n")

        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main ()
