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


def binarize (img, threshold):
    img = np.where(img < threshold, 0, 1)
    img = img.astype(np.float32)

    return img


def label(img, largura_min, altura_min, n_pixels_min):
    img = cv2.convertScaleAbs(img)
    num_labels, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8, ltype=cv2.CV_32S)

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
        median_size = np.median([abs((component['R'] - component['L']) * (component['B'] - component['T'])) for component in components])
        print('Tamanho mediano: %f' % median_size)

        median_width = np.median([component['R'] - component['L'] for component in components])
        print('Largura mediana: %f' % median_width)

        median_height = np.median([component['B'] - component['T'] for component in components])
        print('Altura mediana: %f' % median_height)

        additional_components = 0
        for component in components:
            if component['n_pixels'] > median and abs((component['R'] - component['L']) * (component['B'] - component['T'])) > median_size:
                factor = math.floor(component['n_pixels'] / median)
                size_factor = math.floor(abs((component['R'] - component['L']) * (component['B'] - component['T'])) / median_size)
                width_factor = math.floor((component['R'] - component['L']) / median_width)
                height_factor = math.floor((component['B'] - component['T']) / median_height)

                # additional_components += max(size_factor, factor) - 1
                # additional_components += factor - 1
                additional_components += math.ceil((size_factor + factor) / 2) - 1

                if size_factor > 1:
                    print(size_factor, factor, width_factor, height_factor)

        print('Componentes adicionais: %d' % additional_components)
        print('Componentes totais: %d' % (len(components) + additional_components))
        print("\n--------------------\n")

        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main ()
