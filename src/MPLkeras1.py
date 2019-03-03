import numpy as np
import os
import cv2
import random
import pickle
import mlpfordataset as mymlp
from sklearn.metrics import f1_score
import matplotlib.pyplot as mpl
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD


a = input("Enter 1 for cat-dog, 2 for mnist")
if (int(a) == 1):
    datadir = "/storage2/home2/e1-313-15521/tipr-second-assignment/data/Cat-Dog/"
    #datadir = '../data1/Cat-Dog'
    categs = ["cat", "dog"]
    categslabel = ["0","1"]
else:
    datadir = "/storage2/home2/e1-313-15521/tipr-second-assignment/data/MNIST/"
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
        if random.random() < 0.7:
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


batch_size = 256
num_classes = 10 #10 for mnist, 2 for cd
epochs = 100


x_train = X2
x_test = X2test
y_train = y11
y_test = y11test

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(30, activation='sigmoid', input_shape=(784,))) #784 for mnist, 40000 for cd
#model.add(Dense(30, activation='sigmoid'))
#model.add(Dense(30, activation='sigmoid'))
#model.add(Dense(30, activation='sigmoid'))
#model.add(Dense(30, activation='sigmoid'))
model.add(Dense(10, activation='softmax')) #10 for mnist, 2 for cd

model.summary()
sgd = SGD(lr=0.1,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, #RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
                    
score = model.evaluate(x_test, y_test, verbose=0)

print('Test accuracy:', score[1])
