import numpy as np


def svd(matrix):
    matrix_transpose = np.transpose(matrix)

    aat = np.matmul(matrix, matrix_transpose)

    aat_evs, aat_evcs = np.linalg.eig(aat)

    idx = aat_evs.argsort()[::-1]
    aat_evs = aat_evs[idx]
    aat_evcs = aat_evcs[:, idx]

    aat_evs = aat_evs[:np.shape(matrix)[0]]

    d = np.sqrt(aat_evs)

    v = np.zeros(shape=(np.shape(matrix)[1], np.shape(matrix)[0]))

    for i in range(len(aat_evcs)):
        v[:, i] = np.dot(matrix_transpose, aat_evcs[:, i]) / d[i]

    return aat_evcs, d, np.transpose(v)
