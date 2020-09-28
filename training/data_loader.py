import h5py, time
import tensorflow as tf
import numpy as np

def load_data_new(path):
    data = h5py.File(path, mode='r')
    print("Loading...", path)
    edgefeature = data['edgefeature']
    edgenum = len(edgefeature[0])
    modelnum = len(edgefeature[0][0])
    partnum = len(edgefeature[0][0][0])

    edgefeature_x = np.zeros((partnum, modelnum, edgenum, 2)).astype('float32')
    edgefeature_x = edgefeature
    edgefeature_x = np.transpose(edgefeature_x, (3, 2, 1, 0))

    mask_sum = np.sum(edgefeature_x, axis=3)
    mask_sum = np.sum(mask_sum, axis=2)
    mask_x = np.where(mask_sum == 0, mask_sum, 1)

    e_neighbour = data['e_neighbour']
    e_nb = load_neighbour(e_neighbour, edgenum)

    maxdegree = e_nb.shape[1]
    degree = maxdegree
    return mask_x, partnum, modelnum, edgenum, edgefeature_x, e_nb, maxdegree, degree

def load_neighbour(neighbour, edges, is_padding=False):
    data = neighbour

    if is_padding == True:
        x = np.zeros((edges + 1, 4)).astype('int32')

        for i in range(0, edges):
            x[i + 1] = data[:, i] + 1

            for j in range(0, 4):
                if x[i + 1][j] == -1:
                    x[i + 1][j] = 0

    else:
        x = np.zeros((edges, 4)).astype('int32')

        for i in range(0, edges):
            x[i] = data[:, i]

    return x


def load_labelMatrix(path):
    data = h5py.File(path, mode='r')
    labelMatrix = data['labelMatrix']
    labelMatrix = np.transpose(labelMatrix)
    return labelMatrix

def load_structMatrix(path):
    data = h5py.File(path, mode='r')
    structMatrix = data['structMatrix']
    structMatrix = np.transpose(structMatrix)
    return structMatrix