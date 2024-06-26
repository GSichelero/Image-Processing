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


def separated_mean_filter(img, window_size):
    height = img.shape[0]
    width = img.shape[1]
    color_channels = img.shape[2]
    half_window = window_size // 2
    horizontally_filtered_img = np.zeros((height, width, color_channels), np.uint8)
    output_img = np.zeros((height, width, color_channels), np.uint8)

    for h in range(color_channels):
        for i in range(height):
            for j in range(width):
                sum_values = 0
                count_pixels = 0
                for k in range(-half_window, half_window + 1):
                    if j+k >= 0 and j+k < width:
                        sum_values += img[i, j+k, h]
                        count_pixels += 1
                horizontally_filtered_img[i, j, h] = sum_values // count_pixels

    for h in range(color_channels):
        for i in range(height):
            for j in range(width):
                sum_values = 0
                count_pixels = 0
                for k in range(-half_window, half_window + 1):
                    if i+k >= 0 and i+k < height:
                        sum_values += horizontally_filtered_img[i+k, j, h]
                        count_pixels += 1
                output_img[i, j, h] = sum_values // count_pixels

    return output_img


def integral_image_filter(img, window_size):
    def calculate_integral_image(img):
        height = img.shape[0]
        width = img.shape[1]
        color_channels = img.shape[2]
        integral_img = np.zeros((height, width, color_channels), np.uint32)
        for h in range(color_channels):
            for i in range(height):
                for j in range(width):
                    integral_img[i, j, h] = img[i, j, h]
                    if i > 0:
                        integral_img[i, j, h] += integral_img[i-1, j, h]
                    if j > 0:
                        integral_img[i, j, h] += integral_img[i, j-1, h]
                    if i > 0 and j > 0:
                        integral_img[i, j, h] -= integral_img[i-1, j-1, h]

        return integral_img
    
    
    height = img.shape[0]
    width = img.shape[1]
    color_channels = img.shape[2]
    half_window = window_size // 2
    integral_img = calculate_integral_image(img)
    output_img = np.zeros((height, width, color_channels), np.uint8)

    for h in range(color_channels):
        for i in range(height):
            for j in range(width):
                x1 = max(0, i - half_window)
                x2 = min(height - 1, i + half_window)
                y1 = max(0, j - half_window)
                y2 = min(width - 1, j + half_window)
                count_pixels = (x2 - x1 + 1) * (y2 - y1 + 1)
                sum_values = integral_img[x2, y2, h]
                if x1 > 0 and y1 > 0:
                    sum_values += integral_img[x1 - 1, y1 - 1, h]
                if x1 > 0:
                    sum_values -= integral_img[x1 - 1, y2, h]
                if y1 > 0:
                    sum_values -= integral_img[x2, y1 - 1, h]
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

    start_time = timeit.default_timer()
    output_img = separated_mean_filter(img, WINDOW)
    print('Tempo Filtro separável: %f' % (timeit.default_timer() - start_time))
    cv2.imshow('Imagem filtrada Filtro separável', output_img)

    start_time = timeit.default_timer()
    output_img = integral_image_filter(img, WINDOW)
    print('Tempo Algoritmo com imagens integrais: %f' % (timeit.default_timer() - start_time))
    cv2.imshow('Imagem filtrada Algoritmo com imagens integrais', output_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main ()
