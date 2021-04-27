import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random


def ApplyFiter(img: np.array, matrix: np.array, layer: int):

  # takes an array or PIL image object and
  # applys convolutional filter to image layer (0, 1, 2)
  # and returns 3D image matrix of with dimention 2 being
  # of length 1 for concatenation to create image

    k = len(matrix[0])
    if not type(img) == "np.array":
        img = np.array(img)
    img = img[:, :, [layer]]
    dims = img.shape

    img = np.squeeze(img)

    adjustment0 = dims[0] % k
    adjustment1 = dims[1] % k

    for i in range(0, dims[0] - adjustment0, k):
        for j in range(0, dims[1] - adjustment1, k):
            img[i: i + k, j: j +
                k] = np.multiply(img[i: i + k, j: j + k],  matrix)

    return np.expand_dims(img, axis=2)


def UseFillMatrices(img, rValue=1, gValue=1, bValue=1, MatrixSize=10):
    matrixR = np.full((MatrixSize, MatrixSize), rValue)
    matrixG = np.full((MatrixSize, MatrixSize), gValue)
    matrixB = np.full((MatrixSize, MatrixSize), bValue)
    imgR = ApplyFiter(img, matrixR, layer=0)
    imgG = ApplyFiter(img, matrixG, layer=1)
    imgB = ApplyFiter(img, matrixB, layer=2)

    img = np.concatenate((imgR, imgG, imgB), axis=2)
    return img


def SaveImgFromArray(img):
    im = Image.fromarray(img.astype("uint8"))
    im.save("images/" + "%x" % random.getrandbits(128) + ".jpeg")


img = Image.open("1b58ef6e3d36a42e01992accf5c52d6eea244353.jpg")
img = np.array(img)
img = UseFillMatrices(img, 1, 0.5, 5, 10)
SaveImgFromArray(img)
