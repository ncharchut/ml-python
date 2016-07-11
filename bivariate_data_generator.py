import matplotlib.pyplot as plt
import numpy as np
import random


"""Mimicks Gaussian Mixture in the Elements of Statistical 
    Learning, Chapter 2"""


m_0 = []
m_1 = []
mu_0 = [1,0]
mu_1 = [0,1]
cov = [[1,0],[0,1]]

for _ in xrange(10):
    x, y = np.random.multivariate_normal(mu_0,cov,1).T
    m_0.append([x[0],y[0]])
    x, y = np.random.multivariate_normal(mu_1,cov,1).T
    m_1.append([x[0],y[0]])

x_0, y_0 = [], []
x_1, y_1 = [], []
cov = [[1./5,0],[0,1./5]]

np.append(x_0, [1])

for _ in xrange(100):
    m_k = random.choice(m_0)
    x, y =  np.random.multivariate_normal(m_k,cov,1).T
    x_0.append(x[0])
    y_0.append(y[0])
    
    m_k = random.choice(m_1)
    x, y =  np.random.multivariate_normal(m_k,cov,1).T
    x_1.append(x[0])
    y_1.append(y[0])

data_0 = plt.plot(x_0,y_0,'o',color='blue',fillstyle='none')
data_1 = plt.plot(x_1,y_1,'o',color='orange',fillstyle='none')
plt.show()
