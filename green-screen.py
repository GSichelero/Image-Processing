import sys
import cv2
import numpy as np


def main():
    chroma_key_background_img = cv2.imread('pacote-trabalho3/Wind Waker GC.bmp', cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    for image_number in range(0, 9):
        img = cv2.imread(f'trabalho5/img/{image_number}.bmp', cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        original_image = img.copy()
        chroma_key_background_img = cv2.resize(chroma_key_background_img, (img.shape[1], img.shape[0]), chroma_key_background_img)
        if img is None:
            print('Erro abrindo a imagem.\n')
            sys.exit()

        cv2.imshow(f'{image_number} - original', img)

        b, g, r = cv2.split(img)
        
        mask = g - np.maximum(b, r)
        mask[mask < 0] = 0
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        invert_mask = 1 - mask
        green_background = mask >= 0.2
        green_foreground = mask < 0.2

        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        green_background = cv2.GaussianBlur(green_background.astype(np.float32), (5, 5), 0)
        green_foreground = cv2.GaussianBlur(green_foreground.astype(np.float32), (5, 5), 0)

        green_background = green_background[..., np.newaxis]
        green_foreground = green_foreground[..., np.newaxis]
        mask = mask[..., np.newaxis]
        invert_mask = invert_mask[..., np.newaxis]

        background = chroma_key_background_img * mask
        foreground = img * green_foreground

        cv2.imshow(f'{image_number} - background', background)
        cv2.imshow(f'{image_number} - foreground', foreground)

        result = background + foreground

        cv2.imshow(f'{image_number} - chroma key', result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()