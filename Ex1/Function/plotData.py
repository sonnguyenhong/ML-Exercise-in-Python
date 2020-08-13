import matplotlib.pyplot as plt

def plotData(x,y, theta):
    plt.figure()
    y_new = theta[0] + theta[1] * x
    plt.plot(x, y, 'rx', ms=5, mec='k')
    plt.plot(x, y_new, label='Gradient Descent')
    plt.show()