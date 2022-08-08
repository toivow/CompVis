import numpy
import numpy as np


def estimateF(x1, x2):
    """
    :param x1: Points from image 1, with shape (coordinates, point_id)
    :param x2: Points from image 2, with shape (coordinates, point_id)
    :return F: Estimated fundamental matrix
    """

    # Use x1 and x2 to construct the equation for homogeneous linear system
    ##-your-code-starts-here-##
    u = x1[0]
    v = x1[1]

    u_hat = x2[0]
    v_hat = x2[1]

    eq = np.vstack(((u*u_hat), (u_hat*v), (u_hat), (v_hat*u), (v_hat*v), (v_hat), u, v, np.ones(11))).T
    ##-your-code-ends-here-##

    # Use SVD to find the solution for this homogeneous linear system by
    # extracting the row from V corresponding to the smallest singular value.
    ##-your-code-starts-here-##
    u, s, v = np.linalg.svd(eq)
    v_min = v[np.argmin(s)]
    ##-your-code-ends-here-##
    F = np.reshape(v_min, (3, 3))  # reshape to acquire Fundamental matrix F

    # Enforce constraint that fundamental matrix has rank 2 by performing
    # SVD and then reconstructing with only the two largest singular values
    # Reconstruction is done with u @ s @ vh where s is the singular values
    # in a diagonal form.
    ##-your-code-starts-here-##
    u, s, vh = np.linalg.svd(F)
    idxs = np.argsort(s)[-2:]
    s = numpy.array([[s[idxs[1]], 0, 0],[0, s[idxs[0]], 0], [0, 0, 0]])
    F = u @ s @ vh
    ##-your-code-ends-here-##

    return F
