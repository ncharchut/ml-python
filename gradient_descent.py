import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math

############# USING NUMPY ################

def normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    normalized_X = (X-mu)/sigma
    return np.insert(normalized_X, 0, 1, axis=1)

def J_mse(Theta, X, y):
    m = len(y)
    # J = 1/(2*m) * sum((X*Theta - y)^2)
    return 1/(2.*m)*sum(np.square((np.subtract(X.dot(Theta),y))))

def gradient_descent(Theta, X, y, alpha=0.01, num_iterations=1000, live=False):
    m = len(y)
    J_cost = np.zeros((num_iterations,1))
    X = normalize(X)
    Theta_history = [Theta]

    for i in xrange(num_iterations):
        # Theta = Theta - alpha/m * X'* (X * Theta - y)
        Theta = np.subtract(Theta, alpha/m*X.T.dot(np.subtract(X.dot(Theta),y)))
        J_cost[i] = J_mse(Theta, X, y)

        if live:
            Theta_history.append(Theta)

    return Theta, Theta_history, J_cost, X

def sample_gradient_descent():
    xdata, ydata = [], []
    with open('data/ex1data1.txt','rU') as f:
        for row in f:
            x, y1 = row.split(',')
            xdata.append(float(x))
            ydata.append(float(y1))

    data = zip(xdata,ydata)
    xdata, ydata = zip(*sorted(data, key=lambda x: x[0]))
    X = np.array([xdata]).T
    y = np.array([ydata]).T
    Theta = np.zeros((len(X[0])+1,1))
    Theta, Theta_history, J, X = gradient_descent(Theta, X, y,live=True)

    ax = plt.gca()
    ax.set_title('Sample Gradient Descent',fontsize=28)
    ax.plot(X[:,1],y,'x',color='blue',label='Training Set')

    for i in xrange(len(Theta_history)/2):
        line = Theta_history[i]
        x = X[:,1]
        y = X.dot(line)
        reg_line = ax.plot(x, y, color='red',label='Regressed Line')
        leg = ax.legend(fancybox=True,shadow=True,loc=0,framealpha=0.5)
        text = ax.text(x[-1], y[-1], i+1,weight='bold',fontsize=15,bbox=dict(fc='1', ec='1'))
        plt.pause(.001)
        ax.lines.pop()
        text.remove()

    plt.show()

##########################################
