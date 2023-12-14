# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:24:45 2023

@author: szyns
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

#4.1
data = pd.read_csv('voice_extracted_features.csv',sep=',')
columns = list(data.columns)
vals = data.values
x = vals[:,:-1]
y = vals[:,-1]


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,shuffle=True,stratify=y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


pca = PCA(2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)


x_train_pca_scaled = scaler.fit_transform(x_train_pca)
x_test_pca_scaled = scaler.transform(x_test_pca)

model = kNN(5, weights ='uniform')
model.fit(x_train_pca_scaled,y_train)
y_pred = model.predict(x_test_pca_scaled)
cm = confusion_matrix(y_test, y_pred)
displayed = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
displayed.plot()

rcParams['font.family']='Arial'
rcParams['lines.color']='blue'
rcParams['font.size']=20
females = y_train=='female'
males = y_train=='male'
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.scatter(x_train_pca_scaled[females,0],x_train_pca_scaled[females,1],label='female')
ax.scatter(x_train_pca_scaled[males,0],x_train_pca_scaled[males,1],label='male')
ax.legend()
fig.tight_layout()

pcaX = PCA()
x_train_pca = pcaX.fit_transform(x_train)
x_test_pca = pcaX.transform(x_test)

variances = pcaX.explained_variance_ratio_
cumulated_variances = variances.cumsum()
plt.subplots(1,1,figsize=(10,10))
plt.scatter(np.arange(variances.shape[0]),cumulated_variances)
plt.yticks(np.arange(0,1.1,0.1))
PC_num = (cumulated_variances < 0.95).sum()
print("Optymalna liczba cech dla progu 0.95: ",PC_num)


pipe = Pipeline([['transformer',PCA(2)],['scaler',StandardScaler()],['classifier',kNN(19, weights='uniform')]])
pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
cmx= confusion_matrix(y_test, y_pred)
print("Confusion matrix for the pipe-trained model: ",cmx)
displayedx = ConfusionMatrixDisplay(cmx,display_labels=pipe.classes_)
displayedx.plot()



#4.2

models = [kNN(),SVC(),DecisionTreeClassifier()]
macierze = np.array( [[0,0],
            [0,0]])

def uczenie(n):
    for i in range(0,n):
        for model in models:
            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            cm=confusion_matrix(y_test, y_pred)
            print("Confusion matrix for model: ",model,"test no. ",i)
            print(cm)
            macierze[0][0]+=cm[0][0]
            macierze[0][1]+=cm[0][1]
            macierze[1][0]+=cm[1][0]
            macierze[1][1]+=cm[1][1]
    macierze[0][0]=macierze[0][0]/90
    macierze[1][0]=macierze[1][0]/90
    macierze[0][1]=macierze[0][1]/90
    macierze[1][1]=macierze[1][1]/90
    print("\n\n\nUsredniona macierz: ")
    print(macierze)
    display = ConfusionMatrixDisplay(macierze,display_labels=model.classes_)
    display.plot()
    if(((macierze[0][0])/(macierze[0][0]+macierze[0][1]+macierze[1][0]+macierze[1][1]))>((macierze[1][1])/(macierze[0][0]+macierze[0][1]+macierze[1][0]+macierze[1][1]))):
        print("Metody latwiej wykrywaja kobiety")
    elif ((macierze[0][0]/(macierze[0][0]+macierze[0][1]+macierze[1][0]+macierze[1][1]))==((macierze[1][1])/(macierze[0][0]+macierze[0][1]+macierze[1][0]+macierze[1][1]))):
        print("Metody wykrywaja kobiety i mezczyzn tak samo latwo")
    else:
        print("Metody latwiej wykrywaja mezczyzn")

uczenie(30)

#4.3
class PipeAdd():
    def __init__(self,procent=0.95):
        self.procent = procent
        self.num = None
        self.pca = None
        
    def fit(self,x,y=None):
        self.pca = PCA()
        self.pca.fit(x)
        variances = self.pca.explained_variance_ratio_
        cumulated_variances = variances.cumsum()
        self.num = (cumulated_variances < self.procent).sum()
        return self
    
    def transform(self,x):
        return self.pca.transform(x)[:,:self.num]
    
    def fit_transform(self, x, y=None):
       self.fit(x)
       return self.transform(x)
    



pipeX = Pipeline([['pipeAdd',PipeAdd(procent=0.69)],['scaler',StandardScaler()],['classifier',kNN(19, weights='uniform')]])
pipeX.fit(x_train,y_train)
y_predX = pipe.predict(x_test)

    
#4.4
from sklearn.neighbors import LocalOutlierFactor


class OutlierRemoval:
    def __init__(self):
        self.lof = LocalOutlierFactor()

    def fit(self, X, y=None):
        self.outliers = self.lof.fit_predict(X)
        return self

    def transform(self, X):
        return X[self.outliers != -1, :]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def get_outliers(self):
        return self.outliers

    def get_lof_scores(self, X):
        return -self.lof.negative_outlier_factor_



pipeline_outlier_removal = Pipeline([
    ('outlier_removal', OutlierRemoval()),
    ('scaler', StandardScaler()),
    ('classifier', kNN(n_neighbors=5, weights='distance'))
])
