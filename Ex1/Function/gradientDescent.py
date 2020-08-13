import numpy as np
from Function.computeCost import computeCost
def gradientDescent(X, y, theta, alpha, num_iters):
    m = y.size
    #J_history: update J for each iteration
    J_history = np.zeros(num_iters)

    for i in np.arange(num_iters):
        temp = np.dot(X, theta)
        theta = theta - (alpha/m) * (np.dot(X.T, temp-y))
        J_history[i] = computeCost(X, y, theta)
    return (theta, J_history)