'''
predict profits depend on population
'''

import numpy as np
from Function.plotData import plotData
from Function.computeCost import computeCost
from Function.gradientDescent import gradientDescent
import matplotlib.pyplot as plt
from Function.predict import predict
data = np.loadtxt('ex1data1.txt', delimiter=',')
x = np.array(data[:, 0]).T
y = np.array([data[:, 1]]).T

ones = np.array(np.ones(data.shape[0])).T

X = np.stack((ones, x), axis=1)

theta = np.zeros([2, 1])
iteration = 1500
alpha = 0.01
[theta, J_history] = gradientDescent(X, y, theta, alpha, iteration)

print(predict(30, theta))
plotData(x, y, theta)