import kNN
import matplotlib
import matplotlib.pyplot as plt

datingDataMat,datingLabels = kNN.file2matrix("test.txt")
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()

# group,labels = kNN.createDataSet()
# 
# print(group)
# print(labels)
# 
# print(kNN.classify0([0.0,1.1], group, labels, 3))