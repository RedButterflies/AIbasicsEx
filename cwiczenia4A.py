# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:23:06 2023

@author: szyns
"""
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping

data = load_digits()
x = data.data
y = data.target
y = pd.Categorical(y)
y = pd.get_dummies(y).values

class_num = y.shape[1]

#Podzial danych z uzyciem KFold(5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)


liczba_warstw = [3]#, 5]#,6, 7, 8, 9]
liczba_neuronow = [16]#, 32]#, 48, 64, 80, 96]
funkcja_aktywacji = ['tanh', 'softsign']#, 'softmax', 'relu', 'elu', 'selu']
rozmiar_porcji = [16, 32]#, 64, 80, 96]
optymalizator = [Adam]#, RMSprop, SGD]
predkosc_nauczania = [0.0001, 0.0003]#, 0.0009, 0.0027, 0.0081, 0.0243]

scaler = StandardScaler()

best_accuracy = 0.0
best_hyperparameters = {}

for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    x_train_walidacja_krzyzowa = scaler.fit_transform(x_train)
    x_test_walidacja_krzyzowa = scaler.transform(x_test)

    for warstwy in liczba_warstw:
        for neurony in liczba_neuronow:
            for aktywacja in funkcja_aktywacji:
                for porcja in rozmiar_porcji:
                    for predkosc in predkosc_nauczania:
                        for opt in optymalizator:
                            model = Sequential()
                            model.add(Dense(64, input_shape=(x.shape[1],), activation='relu'))

                            for i in range(warstwy):
                                model.add(Dense(neurony, activation=aktywacja))

                            model.add(Dense(class_num, activation='softmax'))  # Ostatnia warstwa

                            model.compile(optimizer=opt(learning_rate=predkosc), loss='categorical_crossentropy',
                                          metrics=['accuracy'])

                            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                            history = model.fit(x_train_walidacja_krzyzowa, y_train, batch_size=porcja, epochs=10,
                                                validation_data=(x_test_walidacja_krzyzowa, y_test),
                                                callbacks=[early_stopping], verbose=2)

                            y_pred = model.predict(x_test_walidacja_krzyzowa).argmax(axis=1)
                            y_test_walidacja_krzyzowa = y_test.argmax(axis=1)
                            current_accuracy = accuracy_score(y_test_walidacja_krzyzowa, y_pred)

                            if current_accuracy > best_accuracy:
                                best_accuracy = current_accuracy
                                best_hyperparameters = {
                                    'warstwy': warstwy,
                                    'neurony': neurony,
                                    'aktywacja': aktywacja,
                                    'porcja': porcja,
                                    'predkosc': predkosc,
                                    'opt': opt.__name__
                                }

# Wyswietlenie hiperparametrow i najlepszej dokladnosci
print("Best Hyperparameters:", best_hyperparameters)
print("Best Accuracy:", best_accuracy)