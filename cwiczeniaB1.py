# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 22:23:27 2023

@author: szyns
"""
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
loadeddata = load_breast_cancer()
data = pd.DataFrame(loadeddata.data, columns=loadeddata.feature_names)
values = data.values
column_names = list(data.columns)

data['target'] = loadeddata.target
x = data.drop(columns = 'target').values
y = data['target'].values


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4, shuffle=True)

#model tree
from sklearn.tree import DecisionTreeClassifier as DT
model = DT(max_depth=5)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("Decision tree confusion matrix: ")
print(cm)

#tree plot
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
plt.figure(figsize=(60,30))
plot = plot_tree(model,feature_names=column_names,class_names=['Nie','Tak'],fontsize=20)


from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
x_train_standard = x_train.copy()
x_test_standard = x_test.copy()


standard = StandardScaler()
standard.fit(x_train_standard)
x_train_standard=standard.transform(x_train_standard)
x_test_standard = standard.transform(x_test_standard)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
kNN = KNeighborsClassifier(weights='uniform',algorithm='ball_tree',n_neighbors=10,)
svc = SVC(kernel='poly')

print("\n\n\n")
print("kNN standard scaler: ")
kNN.fit(x_train_standard,y_train)
y_pred = kNN.predict(x_test_standard)
cma = confusion_matrix(y_test, y_pred)
print(cma)
displayeda = ConfusionMatrixDisplay(cma,display_labels=kNN.classes_)
displayeda.plot()
sensitivity = cm[0][0]/(cm[0][0]+cm[0][1])
print("Sensitivity: ", sensitivity)
precision = cm[0][0]/(cm[0][0]+cm[1][0])
print("Precision: ",precision)
specificity = cm[1][1]/(cm[1][0]+cm[1][1])
print("Specificity: ",specificity)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy: ",accuracy)
if((sensitivity+precision)>0):
    F1 = 2*(sensitivity*precision)/(sensitivity+precision)
    print("F1: ",F1)


print("\n\n\n")
print("svc standard scaler: ")
svc.fit(x_train_standard,y_train)
y_pred = svc.predict(x_test_standard)
cmb=confusion_matrix(y_test,y_pred)
print(cmb)
displayedb = ConfusionMatrixDisplay(cmb,display_labels=svc.classes_)
displayedb.plot()
print("Sensitivity: ", sensitivity)
precision = cm[0][0]/(cm[0][0]+cm[1][0])
print("Precision: ",precision)
specificity = cm[1][1]/(cm[1][0]+cm[1][1])
print("Specificity: ",specificity)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy: ",accuracy)
if((sensitivity+precision)>0):
    F1 = 2*(sensitivity*precision)/(sensitivity+precision)
    print("F1: ",F1)



x_train_minmax = x_train.copy()
x_test_minmax = x_test.copy()
minmax= MinMaxScaler()
minmax.fit(x_train_minmax)
x_train_minmax = minmax.transform(x_train_minmax)
x_test_minmax = minmax.transform(x_test_minmax)


print("\n\n\n")
print("kNN minmax scaler: ")
kNN.fit(x_train_minmax,y_train)
y_pred = kNN.predict(x_test_minmax)
cmc = confusion_matrix(y_test, y_pred)
displayedc = ConfusionMatrixDisplay(cmc, display_labels=kNN.classes_)
displayedc.plot()
print(cmc)
print("Sensitivity: ", sensitivity)
precision = cm[0][0]/(cm[0][0]+cm[1][0])
print("Precision: ",precision)
specificity = cm[1][1]/(cm[1][0]+cm[1][1])
print("Specificity: ",specificity)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy: ",accuracy)
if((sensitivity+precision)>0):
    F1 = 2*(sensitivity*precision)/(sensitivity+precision)
    print("F1: ",F1)


print("\n\n\n")
print('scv minmax scaler: ')
svc.fit(x_train_minmax,y_train)
y_pred = svc.predict(x_test_minmax)
cmd = confusion_matrix(y_test,y_pred)
displayed = ConfusionMatrixDisplay(cmd, display_labels=svc.classes_)
displayed.plot()
print(cmd)
print("Sensitivity: ", sensitivity)
precision = cm[0][0]/(cm[0][0]+cm[1][0])
print("Precision: ",precision)
specificity = cm[1][1]/(cm[1][0]+cm[1][1])
print("Specificity: ",specificity)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy: ",accuracy)
if((sensitivity+precision)>0):
    F1 = 2*(sensitivity*precision)/(sensitivity+precision)
    print("F1: ",F1)     



x_train_robust = x_train.copy()
x_test_robust = x_test.copy()
print("\n\n\n")
print("kNN robust scaler: ")
kNN.fit(x_train_robust,y_train)
y_pred = kNN.predict(x_test_robust)
cme = confusion_matrix(y_test,y_pred)
print(cme)
displayede = ConfusionMatrixDisplay(cme,display_labels=kNN.classes_)
displayede.plot()
print("Sensitivity: ", sensitivity)
precision = cm[0][0]/(cm[0][0]+cm[1][0])
print("Precision: ",precision)
specificity = cm[1][1]/(cm[1][0]+cm[1][1])
print("Specificity: ",specificity)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy: ",accuracy)
if((sensitivity+precision)>0):
    F1 = 2*(sensitivity*precision)/(sensitivity+precision)
    print("F1: ",F1)  
    
    
    
print("\n\n\n")
print("svc robust scaler: ")
svc.fit(x_train_robust,y_train)
y_pred = svc.predict(x_test_robust)
cmf = confusion_matrix(y_test,y_pred)
print(cmf)
displayedf = ConfusionMatrixDisplay(cmf,display_labels=svc.classes_)
displayedf.plot()
print("Sensitivity: ", sensitivity)
precision = cm[0][0]/(cm[0][0]+cm[1][0])
print("Precision: ",precision)
specificity = cm[1][1]/(cm[1][0]+cm[1][1])
print("Specificity: ",specificity)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print("Accuracy: ",accuracy)
if((sensitivity+precision)>0):
    F1 = 2*(sensitivity*precision)/(sensitivity+precision)
    print("F1: ",F1)  
    

for feature,importance in zip(loadeddata.feature_names,model.feature_importances_):
    if(importance==0):
        print(feature,": cecha niewazna")
    else:
        print(feature,": cecha wazna, waznosc cechy: ",importance)