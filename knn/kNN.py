from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def createFile(filename):
    fr = open(filename, "w")
    for i in range(1000):
        t = random.randint(1,4)
        if(t == 0):
            fr.write(str(random.randint(23000,50000)) + "\t" + str(absolute(random.random())) + "\t" + str(absolute(random.random()*10)) + "\t" + str(t) + "\r")
        elif(t == 1):
            fr.write(str(random.randint(8000,25000)) + "\t" + str(absolute(random.random())) + "\t" + str(absolute(random.random())*10) + "\t" + str(t) + "\r")
        elif(t == 2):
            fr.write(str(random.randint(500,10000)) + "\t" + str(absolute(random.random())) + "\t" + str(absolute(random.random())*10) + "\t" + str(t) + "\r")

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    print("DataSetSize => ", dataSetSize)
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    print("Tile => ", tile(inX, (dataSetSize,1)))
    sqDiffMat = diffMat**2
    print("sqDiffMat => ", sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]