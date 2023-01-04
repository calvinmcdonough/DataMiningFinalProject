# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random 


def main():
    trainSet = pd.read_csv("finalTrainning.csv")
    testSet = pd.read_csv("finalTesting.csv")
    sm = pd.read_csv("sm.csv")
    
    arrayTrainSet = np.array(trainSet)
    numColumns = len(trainSet.columns)
    smArray = np.array(sm)
    
    print(smArray)
    
    #x_train, x_test, y_train, y_test = train_test_split(arrayTrainSet.data, arrayTrainSet.target, test_size=0.2, random_state=0)
    
    attributes = np.array(arrayTrainSet[:,0:numColumns-1])
    values = np.array(arrayTrainSet[:,numColumns-1])
    
    param = {
        'max_depth': 100,
        'eta':.9,
        'objective': 'multi:softmax',
        'num_class':10}
    
    epochs = 1000
    percent = 0.2
    
    array = []
    randomState=0

    for i in range(15):
        epochs = epochs +1
        randomState+=1
        x_train, x_test, y_train, y_test = train_test_split(attributes, values, test_size=.001, random_state= randomState)
    
        train = xgb.DMatrix(x_train, y_train)
        test = xgb.DMatrix(x_test, y_test)
        actualTest = xgb.DMatrix(np.array(testSet))
      
        bst = xgb.train(param, train, epochs)
        #predictions = bst.predict(test)
        predictions = bst.predict(actualTest)
        #print(accuracy_score(y_test,predictions))
        
        array.append(predictions)
    #print(array)
    a = majority_vote(array)
    make_submission(smArray, a)
    #print(accuracy_score(y_test,a))
def majority_vote(array):
    returnArray = []
    for i in range(len(array[0])):
        vote1=0
        vote0=0
        for j in range(len(array)):
            if array[j][i] == 0:
                vote0 += 1
            else:
                vote1 += 1
        if vote1 > vote0:
            returnArray.append(1)
        else:
            returnArray.append(0)
    return returnArray                

def make_submission(smArray,predictions):
   for i in range(len(smArray)):
       predictions[i]=int(predictions[i])
       smArray[i][1]= predictions[i]
       #print(smArray[i][1]," ", predictions[i])
       #print(smArray[i][0],smArray[i][1])
    
   pd.DataFrame(smArray).to_csv("submit.csv")
    
   
    
main()