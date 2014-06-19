import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import all
import operator
from array import array

group,labels = kNN.createDataSet()

print(kNN.classify0([1.0,1.0], group, labels, 3))

datingDataMat = array([[72917, 7.106, .2236],[14283, 2.441, .1908],[73457, 8.3101, .8527],[12429, 4.4323, .9246]])
# datingDataMat = zeros((3,3))

print(datingDataMat)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()