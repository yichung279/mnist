import numpy as np
import cv2

def arr2img(arr, filename):
    img = []
    row = []
    for i, label in enumerate(arr):
        if label == 0:
            pixel = [0, 0, 0]
        elif label == 1:
            pixel = [255, 255, 0]
        elif label == 2:
            pixel = [255, 150, 0]
        elif label == 3:
            pixel = [255, 0, 38]
        elif label == 4:
            pixel = [0, 255, 0]
        elif label == 5:
            pixel = [3, 199, 0]
        elif label == 6:
            pixel = [1, 149, 0]
        elif label == 7:
            pixel = [0, 255, 253]
        elif label == 8:
            pixel = [3, 199, 255]
        elif label == 9:
            pixel = [0, 120, 255]
        elif label == 10:
            pixel = [0, 0, 255]
        elif label == 11:
            pixel = [0, 0, 200]
        elif label == 12:
            pixel = [0, 0, 145]
        elif label == 13:
            pixel = [223, 3, 223]
        elif label == 14:
            pixel = [185, 8, 109]


        row.append(pixel)
        if (i + 1) % 288 == 0:
            img.append(row)
            row = []
    img = np.array(img)
    img = np.rot90(img)
    img = cv2.flip(img, 0)
    cv2.imwrite(filename, img)

if __name__ == '__main__':
    data = np.load('npy1/dataEven_1.02.npy')
    img1 = data[0][0]
    arr2img(img1,'test.jpg')
