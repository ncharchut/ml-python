import matplotlib.pyplot as plt
import numpy as np
import random

m_0 = []
m_1 = []
mu_0 = [1,0]
mu_1 = [0,1]
cov = [[1,0],[0,1]]

for i in xrange(10):
    x, y = np.random.multivariate_normal(mu_0,cov,1).T
    m_0.append([x[0],y[0]])
    x, y = np.random.multivariate_normal(mu_1,cov,1).T
    m_1.append([x[0],y[0]])


x_0, y_0 = [], []
x_1, y_1 = [], []
cov = [[1./5,0],[0,1./5]]

np.append(x_0, [1])

for j in xrange(100):
    m_k = random.choice(m_0)
    x, y =  np.random.multivariate_normal(m_k,cov,1).T
    x_0.append(x[0])
    y_0.append(y[0])
    
    m_k = random.choice(m_1)
    x, y =  np.random.multivariate_normal(m_k,cov,1).T
    x_1.append(x[0])
    y_1.append(y[0])


data_0 = plt.plot(x_0,y_0,'.',color='blue')
data_1 = plt.plot(x_1,y_1,'x',color='red')
plt.show()


def binary_data():
    return (x_0,y_0),(x_1,y_1)