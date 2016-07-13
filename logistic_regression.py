import numpy as np
from gradient_descent import normalize
from scipy.optimize import fmin_bfgs
import binary_data_generator as bin_data

def sigmoid(X):
    return (1+np.exp(-X))**(-1)

def J_log_regularized(theta, X, y, lamb=0):
    m = len(y)
    hx = sigmoid(X.dot(theta))
    J = 1./m*(-y.T.dot(np.log(hx)) - (1 - y).T.dot(np.log(1-hx))) +\
        lamb/(2.*m) * sum(theta**2)
    return J[0]

def plot_decision_boundary(plt, Theta, X, y):
    if np.size(X,1) <= 2:
        plot_x = [min(X[:,1])-2, max(X[:,1])+2]
        slope = -1./Theta[2]*Theta[1]
        plot_y = [slope*plot_x[0] + Theta[0], slope*plot_x[1] + Theta[0]]
        plt.plot(plot_x,plot_y,color='r',label='Decision Boundary')
    plt.show()

def log_gradient_descent(X, y, alpha=0.01, num_iterations=1000, live=False):
    m = len(y)
    J_cost = np.zeros((num_iterations,1))
    X = normalize(X)
    Theta = np.zeros((len(X[0]),1))
    args_l = (X, y)

    return fmin_bfgs(J_log_regularized, Theta, args=args_l)

data = bin_data.sample_binary_data
(x_0,y_0),(x_1,y_1) = data()
xdata = []
ydata = []
for i in xrange(len(x_0)):
    xdata.append([x_0[i],y_0[i]])
    ydata.append([0])
    xdata.append([x_1[i],y_1[i]])
    ydata.append([1])

new_data = zip(xdata,ydata)
xdata, ydata = zip(*sorted(new_data, key=lambda x: x[0][0]))
X = np.array(xdata)
y = np.array(ydata)

theta_opt = log_gradient_descent(X,y)

plt = bin_data.plot_data((x_0,y_0),(x_1,y_1), boundary=True, Theta=theta_opt, X=X, y=y)
