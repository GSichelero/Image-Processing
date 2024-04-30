import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# INPUT_IMAGES =  ['trabalho4/150.bmp', 'trabalho4/205.bmp']

INPUT_IMAGES =  ['trabalho4/60.bmp', 'trabalho4/82.bmp', 'trabalho4/114.bmp', 'trabalho4/150.bmp', 'trabalho4/205.bmp']

# BLOCK_SIZE = 1001 or 1001
# C = -45 or -91

BLOCK_SIZE = 1001
C = -65
MORPHOLOGIC_KERNEL_SIZE = 6

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

        # Abertura
        opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, np.ones((MORPHOLOGIC_KERNEL_SIZE, MORPHOLOGIC_KERNEL_SIZE), np.uint8))
        # cv2.imshow('Imagem aberta', opened_img)

        # Rotulagem
        opened_img = binarize(opened_img, 128)
        cv2.imshow('Imagem binarizada', opened_img)
        components = label(opened_img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
        print('Componentes encontrados: %d' % len(components))

        # Desenha os retângulos
        for component in components:
            cv2.rectangle(img, (component['L'], component['T']), (component['R'], component['B']), (0, 255, 0), 1)

        cv2.imshow('Imagem com componentes', img)



        pixels = [component['n_pixels'] for component in components]
        # plt.hist(pixels, bins=100)
        # plt.xlabel('Número de pixels')
        # plt.ylabel('Número de componentes')
        # plt.title('Distribuição do número de pixels nos componentes')
        # plt.show()
        
        # get the mean number of pixels, the standard deviation and the median, and how many components have more than the 1, 2 and 3 times the standard deviation
        pixels = np.array(pixels)
        mean = np.mean(pixels)
        std = np.std(pixels)
        median = np.median(pixels)
        # print('Média: %f' % mean)
        # print('Desvio padrão: %f' % std)
        print('Mediana: %f' % median)
        # print('Componentes com mais de 1 desvio padrão: %d' % len(pixels[pixels > mean + std]))
        # print('Componentes com mais de 2 desvios padrão: %d' % len(pixels[pixels > mean + 2*std]))
        # print('Componentes com mais de 3 desvios padrão: %d' % len(pixels[pixels > mean + 3*std]))

        # get the mean size of the components
        # mean_size = np.mean([(component['R'] - component['L']) * (component['T'] - component['B']) for component in components])
        # print('Tamanho médio: %f' % mean_size)
        median_size = np.median([abs((component['R'] - component['L']) * (component['T'] - component['B'])) for component in components])
        print('Tamanho mediano: %f' % median_size)

        additional_components = 0
        for component in components:
            if component['n_pixels'] > median and abs((component['R'] - component['L']) * (component['T'] - component['B'])) > median_size:
                factor = math.floor(component['n_pixels'] / median)
                size_factor = math.floor(abs((component['R'] - component['L']) * (component['T'] - component['B'])) / median_size)
                additional_components += factor + math.floor(math.sqrt(size_factor)) - 2

        print('Componentes adicionais: %d' % additional_components)
        print('Componentes totais: %d' % (len(components) + additional_components))
        print("--------------------\n")

        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main ()
