import numpy as np
import scipy


Xdata, ydata = [], []

with open('data/ex1data1.txt','rU') as f:
	for row in f:
		x, y1 = row.split(',')
		Xdata.append(float(x))
		ydata.append(float(y1))

X = np.array([Xdata]).transpose()
y = np.array([ydata]).transpose()
Theta = np.zeros((len()))

############# USING NUMPY ################

def mean_square_error(Theta, X, y):
	m = len(y)
	return 1/(2.*m)*sum(np.square((np.subtract(X.dot(Theta),y))))

print mean_square_error(Theta,X,y)

def gradient_descent(Theta, X, y, alpha=0.0001, num_iterations=1000):
	m = len(y)
	J_cost = np.zeros((num_iterations,1))

	for i in xrange(num_iterations):
		Theta = np.subtract(Theta, alpha/m*X.transpose().dot(np.subtract(X.dot(Theta),y)))
		J_cost[i] = mean_square_error(Theta, X, y)

	return Theta, J_cost

theta, J = gradient_descent(Theta, X, y)

print theta
print J


##########################################