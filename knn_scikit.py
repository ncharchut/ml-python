from sklearn.neighbors import KNeighborsClassifier
import binary_data_generator as bin_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

data = bin_data.sample_binary_data
(x_0,y_0),(x_1,y_1) = data()
xdata = []
ydata = []
for i in xrange(len(x_0)):
    xdata.append([x_0[i],y_0[i]])
    ydata.append(0)
    xdata.append([x_1[i],y_1[i]])
    ydata.append(1)

new_data = zip(xdata,ydata)
xdata, ydata = zip(*sorted(new_data, key=lambda x: x[0][0]))
X = np.array(xdata)
y = np.array(ydata).T

k = 1
h = .005  # step size in the mesh

weights = 'uniform'
clf = KNeighborsClassifier(k, weights=weights)
clf.fit(X, y)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.contour(xx, yy, Z)

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("2-Class classification (k = %i, weights = '%s')"
          % (k, weights))

plt.show()
