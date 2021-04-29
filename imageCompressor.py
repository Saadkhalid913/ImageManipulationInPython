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


def GreyScale(img):
    # converts an np image to greyscale
    rows = img.shape[0]
    cols = img.shape[1]

    TargetImg = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            R = img[i][j][0]
            G = img[i][j][1]
            B = img[i][j][2]

            TargetImg[i][j] = 0.299*R + 0.587 * G + 0.114*B

    TargetImg = np.expand_dims(TargetImg, axis=2)

    return np.concatenate((TargetImg, TargetImg, TargetImg), axis=2)


def Convolution(img: np.array, matrix: np.array, layer: int):
    # applys some matrix convolution to a given image layer and returns result
    n = len(matrix)
    rows = img.shape[0]
    cols = img.shape[1]

    AdjusmentRows = rows % n
    AdjusmentCols = cols % n

    img = img[:, :, [layer]]
    img = np.squeeze(img, axis=2)  # removing 2nd dim of length 1

    targetImage = np.zeros(img.shape)

    for i in range(len(img) - n):
        for j in range(len(img[0]) - n):
            ElementProduct = np.multiply(img[i: i + n, j: j + n],  matrix)
            WeightedSum = np.sum(ElementProduct) / n**2
            targetImage[i][j] = WeightedSum

    return np.expand_dims(targetImage, axis=2)


def compress(img: np.array, factor: int, layer: int):
    # compresses given layer of an image by some factor
    img = np.squeeze(img[:, :, [layer]], axis=2)

    rows = img.shape[0]
    cols = img.shape[1]

    target = np.zeros((rows // factor, cols // factor))

    for i in range(0, rows - factor, factor):
        for j in range(0, cols - factor, factor):
            WeightedAverage = np.sum(
                img[i: i + factor, j: j + factor]) / (factor ** 2)
            target[i // factor][j // factor] = WeightedAverage

    return np.expand_dims(target, axis=2)


def CompressImage(img: np.array, factor: int):
    # compresses image by some factor
    # rType: np.array
    imgR = compress(img, factor, 0)
    imgG = compress(img, factor, 1)
    imgB = compress(img, factor, 2)
    img = np.concatenate((imgR, imgG, imgB), axis=2)
    print(img.shape)

    return img


def SaveImgFromArray(img):
    # takes an image object and saves it to images folder with random hash as name
    im = Image.fromarray(img.astype("uint8"))
    im.save("images/" + "%x" % random.getrandbits(128) + ".jpeg")


img = Image.open("sample_5184×3456.bmp")
img = np.array(img)
img = CompressImage(img, 64)
SaveImgFromArray(img)
