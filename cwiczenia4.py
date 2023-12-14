# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 21:25:58 2023

@author: szyns
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from keras.models import Sequential
from keras.layers import Input,Dense
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


data = load_digits()
x = data.data
y = data.target

y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]


model = Sequential()
model.add(Dense(128, input_shape = (x.shape[1],),activation = 'selu'))
model.add(Dense(128,activation='selu'))
model.add(Dense(128,activation='selu'))
model.add(Dense(128,activation='selu'))
model.add(Dense(128,activation='selu'))
model.add(Dense(class_num,activation = 'softmax'))
learning_rate = 0.0003
model.compile(optimizer = Adam(learning_rate),loss='categorical_crossentropy',metrics=('accuracy'))
model.summary()
plot_model(model,to_file="my_model.png")


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model.fit(x_train,y_train,batch_size=64,epochs = 100, validation_data = (x_test,y_test),verbose = 2)

historia = model.history.history
funkcja_start_train_set = historia['loss']
funkcja_strat_validation_set = historia['val_loss']
dokladnosc_train_set = historia['accuracy']
dokladnosc_validation_set = historia['val_accuracy']

fig,ax = plt.subplots(1,2,figsize=(20,10))
epochs = np.arange(0,100)
ax[0].plot(epochs,funkcja_start_train_set, label = "Funkcja strat dla zbioru treningowego")
ax[0].plot(epochs,funkcja_strat_validation_set, label="Funkcja start dla zbioru testowego")
ax[0].set_title("Funkcja strat")
ax[0].legend()

ax[1].plot(epochs,dokladnosc_train_set, label="Dokladnosc dla zbioru treningowego")
ax[1].plot(epochs, dokladnosc_validation_set, label="Dokladnosc dla zbioru testowego")
ax[1].set_title("Dokładnosc")
ax[1].legend()
##################################################################################
