import sys
import timeit
import numpy as np
import cv2


INPUT_IMAGE =  'pacote-trabalho3/GT2.BMP'
LUMINOSITY_THRESHOLD = 0.7
BOX_BLUR_WINDOW = 2
GAUSSIAN_BLUR_SIGMA = 0.8


def main ():
    img = cv2.imread(INPUT_IMAGE)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    cv2.imshow('Imagem original', img)

    start_time = timeit.default_timer()


    bright_img = np.zeros_like(img, np.float32)
    for i in range(3):
        bright_img[:, :, i] = img[:, :, i] / 255.0
    bright_img = cv2.cvtColor(bright_img, cv2.COLOR_BGR2HSV)
    bright_img[:, :, 2] = np.where(bright_img[:, :, 2] > LUMINOSITY_THRESHOLD, 1.0, 0.0)
    bright_img = cv2.cvtColor(bright_img, cv2.COLOR_HSV2BGR) * 255.0
    bright_img = bright_img.astype(np.uint8)

    cv2.imshow('Imagem brilhante', bright_img)


    gaussian_timer = timeit.default_timer()
    gaussian_images = []
    for i in range(1, 4):
        gaussian_images.append(cv2.GaussianBlur(bright_img, (0, 0), i * GAUSSIAN_BLUR_SIGMA))
    gaussian_output_img = np.zeros_like(bright_img, np.float32)
    for i in range(3):
        gaussian_output_img += gaussian_images[i]
    print('Tempo Filtro Gaussiano: %f' % (timeit.default_timer() - gaussian_timer))
    cv2.imshow('Imagem filtrada Gaussiana', gaussian_output_img)

    box_timer = timeit.default_timer()
    box_blur_images = []
    for i in range(1, 4):
        box_blur_new_img = cv2.blur(bright_img, (BOX_BLUR_WINDOW * i, BOX_BLUR_WINDOW * i))
        box_blur_new_img = cv2.blur(box_blur_new_img, (BOX_BLUR_WINDOW * i, BOX_BLUR_WINDOW * i))
        box_blur_new_img = cv2.blur(box_blur_new_img, (BOX_BLUR_WINDOW * i, BOX_BLUR_WINDOW * i))
        box_blur_images.append(box_blur_new_img)
    box_blur_output_img = np.zeros_like(bright_img, np.float32)
    for i in range(3):
        box_blur_output_img += box_blur_images[i]
    print('Tempo Filtro Box Blur: %f' % (timeit.default_timer() - box_timer))
    cv2.imshow('Imagem filtrada Box Blur', box_blur_output_img)


    gaussian_output_img = cv2.convertScaleAbs(gaussian_output_img)
    gaussian_output_img = cv2.addWeighted(img, 1, gaussian_output_img, 1, 0)
    cv2.imshow('Imagem com efeito Bloom Gaussiano', gaussian_output_img)

    box_blur_bloom_output_img = cv2.convertScaleAbs(box_blur_output_img)
    box_blur_bloom_output_img = cv2.addWeighted(img, 1, box_blur_bloom_output_img, 1, 0)
    cv2.imshow('Imagem com efeito Bloom Box Blur', box_blur_bloom_output_img)


    print('Tempo total: %f' % (timeit.default_timer() - start_time))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main ()