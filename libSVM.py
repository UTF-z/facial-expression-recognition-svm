from svm import *
from os import listdir
from plattSMO import PlattSMO
import pickle
import numpy as np


class LibSVM:

    def __init__(self, data: np.ndarray, label: np.ndarray, C=0, toler=0.0, maxIter=0, **kernelargs):
        self.classlabel = np.unique(label)
        self.classNum = len(self.classlabel)
        self.classifiersNum = (self.classNum * (self.classNum - 1)) / 2
        self.classifiers = []
        self.dataSet = {}
        self.kernelargs = kernelargs
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        m = data.shape[0]
        for i in range(m):
            if label[i] not in self.dataSet.keys():
                self.dataSet[label[i]] = []
                self.dataSet[label[i]].append(data[i, :])
            else:
                self.dataSet[label[i]].append(data[i, :])

    def train(self):
        num = self.classNum
        for i in range(num):
            for j in range(i + 1, num):
                pos = len(self.dataSet[self.classlabel[i]])
                neg = len(self.dataSet[self.classlabel[j]])
                label = np.concatenate((np.ones(pos), -np.ones(neg)))
                data = []
                data.extend(self.dataSet[self.classlabel[i]])
                data.extend(self.dataSet[self.classlabel[j]])
                data = np.array(data)
                svm = PlattSMO(data, label, self.C, self.toler, self.maxIter, **self.kernelargs)
                svm.smoP()
                self.classifiers.append(svm)
        self.dataSet = None

    def predict(self, data: np.ndarray, label: np.ndarray):
        m = shape(data)[0]
        num = self.classNum
        classlabel = []
        err_count = 0.0
        for n in range(m):
            result = np.zeros(num)
            index = -1
            for i in range(num):
                for j in range(i + 1, num):
                    index += 1
                    curr_classifier = self.classifiers[index]
                    pred = curr_classifier.predict([data[n, :]])[0]
                    if pred > 0.0:
                        result[i] += 1
                    else:
                        result[j] += 1
            classlabel.append(result.argmax())
            if classlabel[-1] != label[n]:
                err_count += 1
        return 1.0 - (err_count / m)

    def save(self, filename):
        fw = open(filename, 'wb')
        pickle.dump(self, fw, 2)
        fw.close()

    @staticmethod
    def load(filename):
        fr = open(filename, 'rb')
        svm = pickle.load(fr)
        fr.close()
        return svm


def loadImage(dir, maps=None):
    dirList = listdir(dir)
    data = []
    label = []
    for file in dirList:
        label.append(file.split('_')[0])
        lines = open(dir + '/' + file).readlines()
        row = len(lines)
        col = len(lines[0].strip())
        line = []
        for i in range(row):
            for j in range(col):
                line.append(float(lines[i][j]))
        data.append(line)
        if maps != None:
            label[-1] = float(maps[label[-1]])
        else:
            label[-1] = float(label[-1])
    return data, label


def main():
    data, label = loadImage('trainingDigits')
    svm = LibSVM(data, label, 200, 0.0001, 10000, name='rbf', theta=20)
    svm.train()
    svm.save("svm.txt")
    svm = LibSVM.load("svm.txt")
    test, testlabel = loadImage('testDigits')
    svm.predict(test, testlabel)


if __name__ == "__main__":
    sys.exit(main())
