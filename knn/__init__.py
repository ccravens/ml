import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

kNN.createFile("test.txt")

datingDataMat,datingLabels = kNN.file2matrix("test.txt")
print(datingDataMat)
print(datingLabels)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

# group,labels = kNN.createDataSet()
# 
# print(group)
# print(labels)
# 
# print(kNN.classify0([0.0,1.1], group, labels, 3))