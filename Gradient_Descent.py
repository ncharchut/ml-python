import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Xdata, ydata = [], []
with open('data/ex1data2.txt','rU') as f:
	for row in f:
		x, x1, y1 = row.split(',')
		Xdata.append([float(x),float(x1)])
		ydata.append(float(y1))

X = np.array(Xdata)
y = np.array([ydata]).transpose()
Theta = np.zeros((len(X[0])+1,1))

############# USING NUMPY ################

def normalize(X):
	mu = np.mean(X, axis=0)
	sigma = np.std(X, axis=0)
	normalized_X = (X-mu)/sigma
	return np.insert(normalized_X, 0, 1, axis=1)

def mean_square_error(Theta, X, y):
	m = len(y)
	# J = 1/(2*m) * sum((X*Theta - y)^2)
	return 1/(2.*m)*sum(np.square((np.subtract(X.dot(Theta),y))))

def gradient_descent(Theta, X, y, alpha=0.01, num_iterations=10000):
	m = len(y)
	J_cost = np.zeros((num_iterations,1))
	X = normalize(X)

	for i in xrange(num_iterations):
		# Theta = Theta - alpha/m * X'* (X * Theta - y)
		Theta = np.subtract(Theta, alpha/m*X.transpose().dot(np.subtract(X.dot(Theta),y)))
		J_cost[i] = mean_square_error(Theta, X, y)

	return Theta, J_cost, X

Theta, J, X = gradient_descent(Theta, X, y)
# print Theta
# print J
print X.dot(Theta)
print X

# plt.scatter(X[:,0],y,color='red')
plt.scatter(X[:,1],y,color='blue')
plt.scatter(X[:,1], X.dot(Theta), color='red')
# plt.title('Error over iterations')
# plt.xlabel('Number of Iterations')
# plt.ylabel('Cost J')
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(X[:,1],X[:,2],zs=y,color='blue')

# ax.scatter(X[:,1],X[:,2], zs=Theta.transpose().dot(X.transpose()),color='red')
plt.show()

##########################################