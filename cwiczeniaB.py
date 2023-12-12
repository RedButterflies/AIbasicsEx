# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:41:34 2023

@author: szyns
"""
import pandas as pd

data = pd.read_excel('practice_lab_3.xlsx')


def qualitative_to_0_1X(data,column,value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data
def qualitative_to_0_1(data,column,value_to_be_1):
   data[column]=data[column].apply(lambda x:1 if x==value_to_be_1 else 0)
   return data

def categorical(data,column):
    cat_feature = pd.Categorical(data[column])
    one_hot = pd.get_dummies(cat_feature)
    data = pd.concat([data,one_hot],axis=1)
    data = data.drop(columns = [column])
    return data

data = categorical(data, 'Property_Area')
data = qualitative_to_0_1(data,'Gender', 'Female')
data = qualitative_to_0_1(data, 'Married', 'Yes')
data = qualitative_to_0_1(data, 'Education', 'Graduate')
data = qualitative_to_0_1(data, 'Self_Employed', 'Yes')  
data = qualitative_to_0_1(data, 'Loan_Status', 'Y') 
data = qualitative_to_0_1(data, 'Rural', True) 
data = qualitative_to_0_1(data, 'Semiurban', True)
data = qualitative_to_0_1(data, 'Urban', True)
columns = list(data.columns)
values = data.values
x = data.drop(columns = ['Loan_Status']).values
y = data['Loan_Status'].values



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4, shuffle=True)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
scalers = [StandardScaler(),MinMaxScaler(),RobustScaler()]
for s in scalers:
    print("\n\n\n Scaler: ",s)
    print("\n\n\n")
    scaler = s
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    
    from sklearn.metrics import confusion_matrix
    from sklearn.neighbors import KNeighborsClassifier as kNN
    from sklearn.svm import SVC
    models = [kNN(),SVC()]
    for model in models:
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test,y_pred)
        sensitivity = cm[0][0]/(cm[0][0]+cm[0][1])
        if((cm[0][0]+cm[1][0])>0):
            precision = cm[0][0]/(cm[0][0]+cm[1][0])
        specificity = cm[1][1]/(cm[1][1]+cm[1][0])
        accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
        print(cm)
        print( model,": sensitivity",sensitivity," precision: ",precision,"specificity: ", specificity, "accuracy: ",accuracy)
        if((sensitivity+precision)>0):
            F1 = 2*(sensitivity*precision)/(sensitivity+precision)
            print("F1: ",F1)
    print("\n\n\n")
        
    n_neighbors = [5,10,50]
    weights = ['uniform','distance']
    for x in n_neighbors:
        for i in weights:
            kNNneighbors = kNN(n_neighbors=x,weights=i)
            kNNneighbors.fit(x_train,y_train)
            y_pred = kNNneighbors.predict(x_test)
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            sensitivity = cm[0][0]/(cm[0][0]+cm[0][1])
            if((cm[0][0]+cm[1][0])>0):
               precision = cm[0][0]/(cm[0][0]+cm[1][0])
            specificity = cm[1][1]/(cm[1][1]+cm[1][0])
            accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
            print( "kNN liczba sasiadow: ",x,"sposob wyznaczania wag: ",i,": sensitivity",sensitivity," precision: ",precision,"specificity: ", specificity, "accuracy: ",accuracy)
            if((sensitivity+precision)>0):
                F1 = 2*(sensitivity*precision)/(sensitivity+precision)
                print("F1: ",F1)
    print("\n\n\n")
    kernel = ['rbf','poly','linear','sigmoid']
    for j in kernel:
        SVCa = SVC(kernel=j)
        SVCa.fit(x_train,y_train)
        y_pred = SVCa.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        sensitivity = cm[0][0]/(cm[0][0]+cm[0][1])
        if((cm[0][0]+cm[1][0])>0):
            precision = cm[0][0]/(cm[0][0]+cm[1][0])
        specificity = cm[1][1]/(cm[1][1]+cm[1][0])
        accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
        print( "SVC jadro: ",j,": sensitivity",sensitivity," precision: ",precision,"specificity: ", specificity, "accuracy: ",accuracy)
        if((sensitivity+precision)>0):
            F1 = 2*(sensitivity*precision)/(sensitivity+precision)
            print("F1: ",F1)
    print("\n\n\n")
        
    
    
    
    
    