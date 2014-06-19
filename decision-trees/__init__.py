import trees

myDat,labels=trees.createDataSet()
myDat[0][-1]='maybe'
print(myDat)

print(trees.calcShannonEnt(myDat))