import numpy as np
import os
import cv2
import random
import pickle
import mlpforcv as mymlp
import matplotlib.pyplot as mpl
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import operator
import csv

filename = 'dolphins.csv'
labelfilename = 'dolphins_label.csv'
lines = csv.reader(open(filename))
labelfile = open(labelfilename, 'rt')
lablines = csv.reader(labelfile, delimiter=',')
list1 = list(lablines)
for x1 in range(len(list1)):
    for y1 in range(len(list1[0])):
        list1[x1][y1] = float(list1[x1][y1])
        list1.append(list1[x1])
kl = int(len(list1) / 2)
list1 = list(list1[0:kl])
lines = csv.reader(open(filename, 'rt'), delimiter=' ')
dataset = list(lines)
for x in range(len(dataset)):
    for y in range(len(dataset[0])):
        dataset[x][y] = float(dataset[x][y])
    dataset.append(dataset[x])
k = int(len(dataset) / 2)
dataset = list(dataset[0:k])
X = dataset
label = []
for i in range(len(list1)):
    for j in range(len(list1[0])):
        label.append(list1[i][j])
#print(label)
categslabel = ["0","1","2","3"] #for dolphins; ["0","1","2"] for pubmed; ["0","1","-1"] for twitter

labeltrain = []
trainingSet = []
labeltest = []
testSet = []
for i in range(len(X)):
    if random.random() < 0.7:
        trainingSet.append(X[i]) #issue here
        labeltrain.append(int(label[i]))
    else:
        testSet.append(X[i])
        labeltest.append(int(label[i]))
'''
print(len(dataset))
print(len(dataset[0]))
print(len(trainingSet))
print(len(trainingSet[0]))
print(len(testSet))
print(len(testSet[0]))

'''
newtrain = []
newlabeltrain = []
u = []
for i in range(len(trainingSet)):
    for l1 in range(1000): #make this huge number in actual dataset
        x1 = random.randint(0, len(trainingSet) - 1)
        if (x1 in u):
            continue
        else:
            u.append(x1)
            newtrain.append(trainingSet[x1])
            newlabeltrain.append(labeltrain[x1])
X2 = np.array(newtrain)
y11 = np.zeros(([len(newtrain),1]))
for i in range(len(newtrain)):
    y11[i][0] = newlabeltrain[i]
yl = categslabel
X2test = np.array(testSet)
y11test = labeltest

numlayer1 = input("Enter number of hidden layers")
numneuron1 = list(input("Enter number of neurons in each hidden layer as list")[1:-1].split(" "))
numlayer = int(numlayer1) + 2
print(numlayer)
numneuron = []
numneuron.append(int(len(X2[0]))) #[0]
for i in range(len(numneuron1)):
    numneuron.append(int(numneuron1[i]))
numneuron.append(int(len(yl)))
layeractivfunc1 = list(input("Enter activation function in each hidden layer")[1:-1].split(" "))
print(numneuron)
layeractivfunc = []
layeractivfunc.append("buffer")

for i in range(len(layeractivfunc1)):
    layeractivfunc.append(layeractivfunc1[i])
layeractivfunc.append("softmax")
print(layeractivfunc)

mymlp.mlpmain(numlayer, numneuron, layeractivfunc, X2, y11, X2test, y11test)
