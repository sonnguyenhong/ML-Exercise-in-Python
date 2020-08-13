import numpy as np

def computeCost(X, y, theta):
    m = y.size
    J = 0
    temp = np.dot(X, theta)
    J = (1/(2*m)) * (np.sum(np.array(temp-y) * np.array(temp-y)))
    return J