import kNN
import matplotlib
import matplotlib.pyplot as plt
from numpy import all
import operator
from array import array

datingDataMat,datingLabels = kNN.file2matrix("test.txt")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()
