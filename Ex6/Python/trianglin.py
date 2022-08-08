import numpy as np


def trianglin(P1, P2, x1, x2):
    """
    :param P1: Projection matrix for image 1 with shape (3,4)
    :param P2: Projection matrix for image 2 with shape (3,4)
    :param x1: Image coordinates for a point in image 1
    :param x2: Image coordinates for a point in image 2
    :return X: Triangulated world coordinates
    """
    
    # Form A and get the least squares solution from the eigenvector 
    # corresponding to the smallest eigenvalue
    ##-your-code-starts-here-##

    newx1 = np.array([[0, -x1[2], x1[1]],
                     [x1[2], 0, -x1[0]],
                     [-x1[1], x1[0], 0]])

    newx2 = np.array([[0, -x2[2], x2[1]],
                      [x2[2], 0, -x2[0]],
                      [-x2[1], x2[0], 0]])

    A_top = np.matmul(newx1, P1)
    A_bottom = np.matmul(newx2, P2)

    A = np.vstack((A_top, A_bottom))

    idx = np.argmin(np.linalg.eig(np.matmul(A.T, A))[0])
    X = np.linalg.eig(np.matmul(A.T, A))[1][:, idx]
    ##-your-code-ends-here-##

    #X = np.array([0, 0, 0, 1])  # remove me
    
    return X
