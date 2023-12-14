from keras.layers import Dense, LayerNormalization, BatchNormalization, Dropout, GaussianNoise
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris
import pandas as pd


def zad61():
    data = load_iris()
    y = data.target
    x = data.data
    y = pd.Categorical(y)
    y = pd.get_dummies(y).values
    class_num = y.shape[1]
    reg_rates = [0, 0.0001, 0.001, 0.01, 0.1]
    neuron_num = 64
    do_rate = 0.5
    noise = 0.1
    learning_rate = 0.001
    repeat_num = 3
    
    block = [Dense,]# LayerNormalization, BatchNormalization, Dropout, GaussianNoise]
    args = [(neuron_num, 'selu'), (), (), (do_rate,), (noise,)]
    results= []
    for i in reg_rates:
        accuracies= []
        
        model = Sequential()
        model.add(Dense(neuron_num, activation='relu', input_shape=(x.shape[1],), kernel_regularizer=l2(i)))
        for _ in range(repeat_num):
            for layer, arg in zip(block, args):
                model.add(layer(*arg))
            model.add(Dense(class_num, activation='sigmoid'))
        
            model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
        
            x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, stratify=y)
        
            model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), verbose=2)
        
            accuracies.append(max(model.history.history['val_accuracy']))
        
        results.append(np.mean(accuracies))
    
    plt.scatter(reg_rates,results)
    plt.title('Zaleznosci sredniej od wspolczynnika regularyzacji')
    plt.xscale('log')
    plt.xlabel('Wspolczynnik regularyzacji')
    plt.ylabel('Srednia dokladnosc')
    plt.title('Wplyw wspolczynnika regularyzacji na srednia dokladnosc')
    plt.show()
    return 0
    
def zad62():
        data = load_iris()
        y = data.target
        x = data.data
        y = pd.Categorical(y)
        y = pd.get_dummies(y).values
        class_num = y.shape[1]
        neuron_num = 64
        do_rate = [0,0.2,0.3,0.5]
        noise = 0.1
        learning_rate = 0.001
        repeat_num = 3
        
        block = [Dense,Dropout]# LayerNormalization, BatchNormalization, Dropout, GaussianNoise]
        args = [(neuron_num, 'selu'), (), (), (do_rate,), (noise,)]
        results= []
        for i in do_rate:
            accuracies= []
            
            model = Sequential()
            model.add(Dense(neuron_num, activation='relu', input_shape=(x.shape[1],), kernel_regularizer=l2(0.001)))
            for _ in range(repeat_num):
                model.add(Dense(neuron_num, activation='selu'))
                model.add(Dropout(i))
    
            model.add(Dense(class_num, activation='sigmoid'))
            
            model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, stratify=y)
            
            model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), verbose=2)
            
            accuracies.append(max(model.history.history['val_accuracy']))
            
            results.append(np.mean(accuracies))
        
        plt.scatter(do_rate,results)
        plt.title('Zaleznosci sredniej od parametru do_rate')
        plt.xscale('log')
        plt.xlabel('Parametr do_rate')
        plt.ylabel('Srednia dokladnosc')
        plt.title('Wplyw parametru do_rate na srednia dokladnosc')
        plt.show()
        return 0

def zad63():
        data = load_iris()
        y = data.target
        x = data.data
        y = pd.Categorical(y)
        y = pd.get_dummies(y).values
        class_num = y.shape[1]
        neuron_num = 64
        do_rate = 0.5
        learning_rate = 0.001
        noise = [0, 0.1, 0.2, 0.3]
        
        repeat_num = 3
        
        block = [Dense, GaussianNoise]# LayerNormalization, BatchNormalization, Dropout, GaussianNoise]
        args = [(neuron_num, 'selu'), (), (), (do_rate,), (noise,)]
        results= []
        for i in noise:
            accuracies= []
            
            model = Sequential()
            model.add(Dense(neuron_num, activation='relu', input_shape=(x.shape[1],), kernel_regularizer=l2(0.001)))
            for _ in range(repeat_num):
                model.add(Dense(neuron_num, activation='selu'))
                model.add(GaussianNoise(stddev=i))
    
            model.add(Dense(class_num, activation='sigmoid'))
        
            model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, stratify=y)
            
            model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), verbose=2)
            
            accuracies.append(max(model.history.history['val_accuracy']))
            
            results.append(np.mean(accuracies))
        
        plt.scatter(noise,results)
        plt.title('Zaleznosci sredniej od parametru noise')
        plt.xscale('log')
        plt.xlabel('Parametr noise')
        plt.ylabel('Srednia dokladnosc')
        plt.title('Wplyw parametru noise na srednia dokladnosc')
        plt.show()
        return 0


zad61()
zad62()    
zad63()