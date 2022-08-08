import numpy as np


def camcalibDLT(x_world, x_im):
    """
    :param x_world: World coordinatesm with shape (point_id, coordinates)
    :param x_im: Image coordinates with shape (point_id, coordinates)
    :return P: Camera projection matrix with shape (3,4)
    """

    # Create the matrix A 
    ##-your-code-starts-here-##
    A = 0*np.ones((16, 12))

    row = 0

    for n in range(0, (x_world.shape[0])*2, 2 ):
        X = x_world[row][:]
        X = X.T
        y = x_im[row][1]
        A[n] = np.hstack( (np.zeros(X.shape), X, -y*X ))

        x = x_im[row][0]
        A[n+1] = np.hstack( (X, np.zeros(X.shape), -x*X) )
        row += 1
    ##-your-code-ends-here-##
    
    # Perform homogeneous least squares fitting.
    # The best solution is given by the eigenvector of
    # A.T*A with the smallest eigenvalue.
    ##-your-code-starts-here-##
    idx = np.argmin(np.linalg.eig(np.matmul(A.T, A))[0])
    ev = np.linalg.eig(np.matmul(A.T, A))[1][:, idx]
    ##-your-code-ends-here-##
    
    # Reshape the eigenvector into a projection matrix P
    P = np.reshape(ev, (3, 4))  # here ev is the eigenvector from above
    #P = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=float)  # remove this and uncomment the line above
    
    return P
