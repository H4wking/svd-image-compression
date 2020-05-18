import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
from svd import svd


def img_compr_stacked(image, k):
    img = np.array(Image.open(image))

    img = img / 255
    row, col, _ = img.shape

    img_red = img[:, :, 0]
    img_green = img[:, :, 1]
    img_blue = img[:, :, 2]

    img_rgb = np.hstack((img_red, img_green, img_blue))

    U, D, V = svd(img_rgb)

    bytes_stored = sum([matrix.nbytes for matrix in [U, D, V]])

    U_k = U[:, 0:k]
    V_k = V[0:k, :]
    D_k = D[0:k]

    compressed_bytes = sum([matrix.nbytes for matrix in [U_k, D_k, V_k]])

    img_rgb_compr = np.dot(U_k, np.dot(np.diag(D_k), V_k))

    img_compr = np.zeros((row, col, 3))

    img_red_compr, img_green_compr, img_blue_compr = np.hsplit(img_rgb_compr, 3)

    img_compr[:, :, 0] = img_red_compr
    img_compr[:, :, 1] = img_green_compr
    img_compr[:, :, 2] = img_blue_compr

    img_compr[img_compr < 0] = 0
    img_compr[img_compr > 1] = 1

    plt.imshow(img_compr)
    plt.xlabel('rank = {}'.format(k))

    plt.show()

    matplotlib.image.imsave('compressed_{}.{}'.format(k, image.split('.')[-1]), img_compr)

    mse = ((img - img_compr)**2).mean(axis=None)

    return bytes_stored, compressed_bytes, mse


bytes_stored, compressed_bytes, mse = img_compr_stacked("shrek.jpeg", 100)

print(bytes_stored, compressed_bytes, mse)
