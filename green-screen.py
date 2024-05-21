import sys
import cv2
import numpy as np


def main():
    background_img = cv2.imread('pacote-trabalho3/Wind Waker GC.bmp')
    for image_number in range(0, 1):
        # read the image with the rgb channels between 0 and 1
        img = cv2.imread(f'trabalho5/img/{image_number}.bmp', cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        if img is None:
            print('Erro abrindo a imagem.\n')
            sys.exit()

        cv2.imshow(f'{image_number} - original', img)

        # create a mask for the green comparing the green channel with the biggest of the other two using np.maximum
        b, g, r = cv2.split(img)

        # print b, g, r
        # cv2.imshow(f'{image_number} - b', b)
        # cv2.imshow(f'{image_number} - g', g)
        # cv2.imshow(f'{image_number} - r', r)
        
        # the mask is created by subtracting the green channel from the maximum of the other two
        mask = g - np.maximum(b, r)
        # where the mask is negative, it is set to zero
        mask[mask < 0] = 0
        # normalize the mask to be between 0 and 1
        mask = (mask - mask.min()) / (mask.max() - mask.min())

        # separate the mask in two parts, one for the background and one for the foreground
        # the threshold is 0.2
        green_background = mask >= 0.2
        green_foreground = mask < 0.2

        # cv2.imshow(f'{image_number} - green_background', green_background.astype(np.float32))









        # Separate the mask into background and foreground parts using a threshold of 0.2
        green_background = (mask >= 0.2).astype(np.uint8) * 255  # Convert to 8-bit binary mask
        green_foreground = (mask < 0.2).astype(np.uint8) * 255   # Convert to 8-bit binary mask

        cv2.imshow(f'{image_number} - green_background', green_background)
        cv2.imshow(f'{image_number} - green_foreground', green_foreground)

        # Ensure mask is of type uint8
        mask = (mask * 255).astype(np.uint8)

        # Invert the foreground mask for further operations
        mask_inv = cv2.bitwise_not(green_foreground)

        # Resize the background image to match the size of the foreground image if needed
        height, width = green_foreground.shape
        background_resized = cv2.resize(background_img, (width, height))

        # show the mask and mask_inv
        cv2.imshow(f'{image_number} - mask', mask)
        # Create the masked background and foreground
        background_part = cv2.bitwise_and(background_resized, background_resized, mask=green_background)
        # show the background part
        cv2.imshow(f'{image_number} - background_part', background_part)
        foreground_part = cv2.bitwise_and((img * 255).astype(np.uint8), (img * 255).astype(np.uint8), mask=green_foreground)
        # show the foreground part
        cv2.imshow(f'{image_number} - foreground_part', foreground_part)

        # Combine the background and foreground
        result = cv2.add(background_part, foreground_part)

        # Display the result
        cv2.imshow(f'{image_number} - result', result)



        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()