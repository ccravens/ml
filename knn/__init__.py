import kNN

group,labels = kNN.createDataSet()

print(kNN.classify0([1.0,1.0], group, labels, 3))