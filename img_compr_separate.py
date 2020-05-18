import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
from svd import svd


def img_compr_separate(image, k):
    img = np.array(Image.open(image))

    img = img / 255
    row, col, _ = img.shape

    img_red = img[:, :, 0]
    img_green = img[:, :, 1]
    img_blue = img[:, :, 2]

    U_r, D_r, V_r = svd(img_red)
    U_g, D_g, V_g = svd(img_green)
    U_b, D_b, V_b = svd(img_blue)

    bytes_stored = sum([matrix.nbytes for matrix in [U_r, D_r, V_r, U_g, D_g, V_g, U_b, D_b, V_b]])

    U_r_k = U_r[:, 0:k]
    U_g_k = U_g[:, 0:k]
    U_b_k = U_b[:, 0:k]

    V_r_k = V_r[0:k, :]
    V_g_k = V_g[0:k, :]
    V_b_k = V_b[0:k, :]

    D_r_k = D_r[0:k]
    D_g_k = D_g[0:k]
    D_b_k = D_b[0:k]

    compressed_bytes = sum([matrix.nbytes for matrix in [U_r_k, D_r_k, V_r_k, U_g_k, D_g_k, V_g_k, U_b_k, D_b_k, V_b_k]])

    img_red_compr = np.dot(U_r_k, np.dot(np.diag(D_r_k), V_r_k))
    img_green_compr = np.dot(U_g_k, np.dot(np.diag(D_g_k), V_g_k))
    img_blue_compr = np.dot(U_b_k, np.dot(np.diag(D_b_k), V_b_k))

    img_compr = np.zeros((row, col, 3))

    img_compr[:, :, 0] = img_red_compr
    img_compr[:, :, 1] = img_green_compr
    img_compr[:, :, 2] = img_blue_compr

    img_compr[img_compr < 0] = 0
    img_compr[img_compr > 1] = 1

    plt.xlabel('rank = {}'.format(k))

    plt.imshow(img_compr)
    plt.show()
    matplotlib.image.imsave('compressed_{}.{}'.format(k, image.split('.')[-1]), img_compr)

    mse = ((img - img_compr)**2).mean(axis=None)

    return bytes_stored, compressed_bytes, mse


bytes_stored, compressed_bytes, mse = img_compr_separate("shrek.jpeg", 100)

print(bytes_stored, compressed_bytes, mse)
