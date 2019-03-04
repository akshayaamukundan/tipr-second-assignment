# The entire code should be able to run from this file!
import numpy as np
import os
import cv2
import random
import pickle
import nn as mymlp
from sklearn.metrics import f1_score
import matplotlib.pyplot as mpl
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import sys

for cmd_arg in range(len(sys.argv)):
    if (sys.argv[cmd_arg] == '--train-data'):
        datadir = sys.argv[cmd_arg + 1]
    elif (sys.argv[cmd_arg] == '--test-data'):
        testdatadir = sys.argv[cmd_arg + 1]
    elif sys.argv[cmd_arg] == '--dataset':
        a1 = sys.argv[cmd_arg + 1]
    elif sys.argv[cmd_arg] == '--configuration':
        numlayer11 = sys.argv[cmd_arg + 1:]
    else:
        pass
    
for i in range(len(numlayer11)):
    if(i ==0):
        numlayer2 = numlayer11[0] + " "
    else:
        numlayer2 += numlayer11[i] + " "
numlayer1 = numlayer2[0:-1]     

if (a1 == 'Cat-Dog'):
    #datadir = "/storage2/home2/e1-313-15521/tipr-second-assignment/data1/Cat-Dog/"
    #datadir = '../data1/Cat-Dog'
    categs = ["cat", "dog"]
    categslabel = ["0","1"]
else:
    #datadir = "/storage2/home2/e1-313-15521/tipr-second-assignment/data1/MNIST/"
    #datadir = '../data1/MNIST'
    categs = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    categslabel = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

X = []
label = []
for categ in categs:
    i = 0
    path = os.path.join(datadir,categ)
    label.append(int(categslabel[i]))
    i += 1
    Xlabel = []
    print(path)
    for img in os.listdir(path):
        imgarray = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #colour is not a differentiating factor
        imglinear = []
        for i in range(len(imgarray)):
            for j in range(len(imgarray[0])):
                imglinear.append(imgarray[i][j])
        Xlabel.append(imglinear)
    X.append(Xlabel)

labeltrain = []
trainingSet = []
labeltest = []
testSet = []
for labelclass in range(len(X)):
    for trainclass in range(len(X[int(labelclass)])):
        if random.random() <= 1.0:
            trainingSet.append(X[labelclass][trainclass])
            labeltrain.append(labelclass)
        else:
            testSet.append(X[labelclass][trainclass])
            labeltest.append(labelclass)
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
X2 = np.array(newtrain)/255
y11 = np.zeros(([len(newtrain),1]))
for i in range(len(newtrain)):
    y11[i][0] = newlabeltrain[i]
yl = categslabel


X2test = np.array(testSet)/255
y11test = labeltest

numneuron1 = numlayer1[1:-1].split(" ")
numlayer = int(len(numneuron1)) + 2
print(numlayer)
numneuron = []
numneuron.append(int(len(X2[0]))) #[0]
for i in range(len(numneuron1)):
    numneuron.append(int(numneuron1[i]))
numneuron.append(int(len(yl)))

layeractivfunc = []
layeractivfunc.append("buffer")

for i in range(len(numneuron1)):
    layeractivfunc.append("sigmoid")
layeractivfunc.append("softmax")
print(layeractivfunc)

Xt = []
labelt = []
for categ in categs:
    i = 0
    path = os.path.join(testdatadir,categ)
    labelt.append(int(categslabel[i]))
    i += 1
    Xlabelt = []
    for img in os.listdir(path):
        imgarray = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE) #colour is not a differentiating factor
        imglinear = []
        for i in range(len(imgarray)):
            for j in range(len(imgarray[0])):
                imglinear.append(imgarray[i][j])
        Xlabelt.append(imglinear)
    Xt.append(Xlabelt)

labeltest = []
testSet = []
for labelclass in range(len(Xt)):
    for trainclass in range(len(Xt[int(labelclass)])):
        testSet.append(Xt[labelclass][trainclass])
        labeltest.append(labelclass)

yl = categslabel
X2test = np.array(testSet)/255
y11test = labeltest

mymlp.mlpmain(numlayer, numneuron, layeractivfunc, X2, y11, X2test, y11test)
