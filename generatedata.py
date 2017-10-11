import numpy as np
import matplotlib.pyplot as plt



def createTwoClusters(mean1,mean2,cov1,cov2):
    x1, y1 = np.random.multivariate_normal(mean1, cov1, 500).T
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 500).T
    return x1,y1,x2,y2

def plotData(x,y):
    plt.plot(x, y, 'x')
    plt.axis('equal')
    plt.show()

mean1 = [0, 0]
cov1 = [[1, 0], [0, 10]]  # diagonal covariance
mean2 = [10, 0]
cov2 = [[5, 0], [0, 1]]  # diagonal covariance
x1,y1,x2,y2=createTwoClusters(mean1,mean2,cov1,cov2)
plotData([x1,x2],[y1,y2])
