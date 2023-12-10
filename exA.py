# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 18:18:23 2023

@author: szyns
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#zad 2.1
data = pd.read_excel('practice_lab_2.xlsx')
macierz_korelacji = data.corr()
print(macierz_korelacji)
fig,ax = plt.subplots(4,4,figsize=(20,20))
ax[0,0].scatter(data['Przestepczosc'],data['MedianowaCena'])
ax[0,0].set_title('Korelacja miedzy przestepczoscia a medianowa cena mieszkania')
ax[0,0].set_xlabel('Przestepczosc')
ax[0,0].set_ylabel('Cena medianowa mieszkania')
fig.tight_layout()

ax[0,1].scatter(data['Podatki'],data['MedianowaCena'])
ax[0,1].set_title('Korelacja miedzy podatkami a medianowa cena mieszkania')
ax[0,1].set_xlabel('Podatki')
ax[0,1].set_ylabel('Cena medianowa mieszkania')
fig.tight_layout()


ax[0,2].scatter(data['PrzyRzece'],data['MedianowaCena'])
ax[0,2].set_title('Korelacja miedzy loaklizacja przy rzece a medianowa cena mieszkania')
ax[0,2].set_xlabel('Przy rzece')
ax[0,2].set_ylabel('Cena medianowa mieszkania')
fig.tight_layout()


ax[0,3].scatter(data['TlenkiAzotu'],data['MedianowaCena'])
ax[0,3].set_title('Korelacja miedzy tlenkami azotu a medianowa cena mieszkania')
ax[0,3].set_xlabel('Tlenki azotu')
ax[0,3].set_ylabel('Cena medianowa mieszkania')
fig.tight_layout()

ax[1,3].scatter(data['LPokojow'],data['MedianowaCena'])
ax[1,3].set_title('Korelacja miedzy liczba pokojow a medianowa cena mieszkania')
ax[1,3].set_xlabel('Liczba pokojow')
ax[1,3].set_ylabel('Cena medianowa mieszkania')
fig.tight_layout()

ax[2,3].scatter(data['WiekMieszkan'],data['MedianowaCena'])
ax[2,3].set_title('Korelacja miedzy wiekiem mieszkania a medianowa cena mieszkania')
ax[2,3].set_xlabel('Wiek mieszkan')
ax[2,3].set_ylabel('Cena medianowa mieszkania')
fig.tight_layout()

ax[3,3].scatter(data['OdleglOdCentrow'],data['MedianowaCena'])
ax[3,3].set_title('Korelacja miedzy odlegloscia od centrum a medianowa cena mieszkania')
ax[3,3].set_xlabel('Odleglosc od centrum')
ax[3,3].set_ylabel('Cena medianowa mieszkania')
fig.tight_layout()

ax[1,1].scatter(data['DostDoMetra'],data['MedianowaCena'])
ax[1,1].set_title('Korelacja miedzy dostepem do metra a medianowa cena mieszkania')
ax[1,1].set_xlabel('Dostep do metra')
ax[1,1].set_ylabel('Cena medianowa mieszkania')
fig.tight_layout()

ax[1,2].scatter(data['NauczUczen'],data['MedianowaCena'])
ax[1,2].set_title('Korelacja miedzy stosunkiem ilosci nauczycieli do liczby uczniow  a medianowa cena mieszkania')
ax[1,2].set_xlabel('Stosunek liczby nauczycieli do liczby uczniow')
ax[1,2].set_ylabel('Cena medianowa mieszkania')
fig.tight_layout()

ax[1,0].scatter(data['AfrAmer'],data['MedianowaCena'])
ax[1,0].set_title('Korelacja miedzy liczba Afroamerykanow a medianowa cena mieszkania')
ax[1,0].set_xlabel('Liczba Afroamerykanow')
ax[1,0].set_ylabel('Cena medianowa mieszkania')
fig.tight_layout()


ax[2,1].scatter(data['PracFiz'],data['MedianowaCena'])
ax[2,1].set_title('Korelacja miedzy liczba zamieszkujacych pracownikow fizycznych a medianowa cena mieszkania')
ax[2,1].set_xlabel('Liczba pracownikow fizycznych')
ax[2,1].set_ylabel('Cena medianowa mieszkania')
fig.tight_layout()

#zad 2.2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
data_names = list(data.columns)
data_values = data.values
x = data_values[:,:-1]
y = data_values[:,-1]

print()

def powtarzaneUczenie(n):
    for i in (0,n):
        linReg = LinearRegression()
        x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True)
        linReg.fit(x_train, y_train)
        y_pred = linReg.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mape
print("")
print("Wartosc mape po powtorzeniu eksperymentu n razy:  " ,powtarzaneUczenie(100))
        
#zad 2.3
#zastapienie outliers srednia
def powtarzaneUczenieMean(n):
    for i in (0,n):
        linReg = LinearRegression()
        x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True)
        outliers = np.abs((y_train-y_train.mean())/y_train.std())>3
        #x_train_no_outliers = x_train[~outliers,:]
        #y_train_no_outliers = y_train[~outliers]
        x_train_mean = x_train.copy()
        x_train[outliers] = x_train.mean()
        y_train_mean = y_train.copy()
        y_train_mean[outliers]=y_train.mean()
        linReg.fit(x_train_mean, y_train_mean)
        y_pred = linReg.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mape
print("")
print("Wartosc mape po powtorzeniu eksperymentu n razy po zastapieniu outliers srednia:  " ,powtarzaneUczenieMean(100))

#usuniecie outliers
def powtarzaneUczenieNoOutliers(n):
    for i in (0,n):
        linReg = LinearRegression()
        x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.3,shuffle=True)
        outliers = np.abs((y_train-y_train.mean())/y_train.std())>3
        x_train_no_outliers = x_train[~outliers,:]
        y_train_no_outliers = y_train[~outliers]
        linReg.fit(x_train_no_outliers, y_train_no_outliers)
        y_pred = linReg.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return mape
print("")
print("Wartosc mape po powtorzeniu eksperymentu n razy po usunieciu outliers:  " ,powtarzaneUczenieNoOutliers(100))


#zad 2.4
nowe_dane = np.stack([x[:,4]*x[:,3]+0.1,
                    x[:,4]*x[:,5]+0.1,
                     x[:,5]*x[:,6]+0.1,
                     x[:,1]*x[:,3]+0.1],axis = -1)
rozszerzone_dane = np.concatenate([x,nowe_dane],axis = -1)

def powtarzaneUczenieZmienioneWartosci(n):
    for i in (0,n):
        linReg = LinearRegression()
        x_train,x_test,y_train,y_test = train_test_split(rozszerzone_dane,y,test_size=0.4,shuffle=True)
        linReg.fit(x_train,y_train)
        y_pred = linReg.predict(x_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
    return mape

print("Wartosc mape po dodaniu zmienionych wartosci: ", powtarzaneUczenieZmienioneWartosci(100))


#zad 2.5
from sklearn.datasets import load_diabetes
dataA = load_diabetes()
dataB = pd.DataFrame(dataA.data, columns = dataA.feature_names)



print("Macierz korelacji: ", dataB.corr())

fig,bx = plt.subplots(1,4,figsize=(20,20))
bx[0].scatter(dataB['age'],dataB['s6'])
bx[0].set_title('Wykres zaleznosci miedzy wiekiem a zmienna zalezna')
bx[0].set_xlabel('wiek')
bx[0].set_ylabel('zmienna zalezna')
fig.tight_layout()


bx[1].scatter(dataB['sex'],dataB['s6'])
bx[1].set_title('Wykres zaleznosci miedzy plcia a zmienna zalezna') 
bx[1].set_xlabel('plec')
bx[1].set_ylabel('zmienna zalezna')
fig.tight_layout()
                                
bx[2].scatter(dataB['bmi'],dataB['s6'])
bx[2].set_title('Wykres zaleznosci miedzy bmi a zmienna zalezna')
bx[2].set_xlabel('bmi')
bx[2].set_ylabel('zmienna zalezna')
fig.tight_layout()

bx[3].scatter(dataB['bp'],dataB['s6'])
bx[3].set_title('Wykres zaleznosci miedzy bp a zmienna zalezna')
bx[3].set_xlabel('bp')
bx[3].set_ylabel('zmienna zalezna')
fig.tight_layout()


x = dataB.values[:,:-1]
y = dataB.values[:,-1]
def trenowanieA(n):
    for i in (0,n):
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,shuffle=True)
        linRegA = LinearRegression()
        linRegA.fit(x_train,y_train)
        y_pred=linRegA.predict(x_test)
    
    mapeA = mean_absolute_percentage_error(y_test, y_pred)
    print("Wynik mape dla ",n,'powtorzen: ', mapeA)

def trenowanieB(n):
    for i in (0,n):
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,shuffle=True)
        outliers = np.abs((y_train-y_train.mean())/y_train.std())>3
        x_train_no_outliers = x_train[~outliers,:]
        y_train_no_outliers = y_train[~outliers]
        linRegA = LinearRegression()
        linRegA.fit(x_train_no_outliers,y_train_no_outliers)
        y_pred=linRegA.predict(x_test)
    
    mapeA = mean_absolute_percentage_error(y_test, y_pred)
    print("Wynik mape dla ",n,'powtorzen, po usunieciu outliers: ', mapeA)

def trenowanieC(n):
    for i in (0,n):
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,shuffle=True)
        outliers = np.abs((y_train-y_train.mean())/y_train.std())>3
        x_train_no_outliers = x_train.copy()
        x_train_no_outliers[outliers] = x_train.mean()
        y_train_no_outliers = y_train.copy()
        y_train_no_outliers[outliers] = y_test.mean()
        linRegA = LinearRegression()
        linRegA.fit(x_train_no_outliers,y_train_no_outliers)
        y_pred=linRegA.predict(x_test)
    
    mapeA = mean_absolute_percentage_error(y_test, y_pred)
    print("Wynik mape dla ",n,'powtorzen, po zastapieniu outliers srednia: ', mapeA)






def trenowanieD(n):
    new_data = np.stack([x[:,3]*x[:,0],x[:,5]*x[:,6],x[:,1]*x[:,5],x[:,0]*x[:,3],x[3,5]*x[:,0]],axis = -1)
    dodatkowy_x = np.concatenate([x,new_data],axis = -1)
    for i in (0,n):
        x_train,x_test,y_train,y_test = train_test_split(dodatkowy_x,y,test_size=0.4,shuffle=True)
        linRegA = LinearRegression()
        linRegA.fit(x_train,y_train)
        y_pred=linRegA.predict(x_test)
    
    mapeA = mean_absolute_percentage_error(y_test, y_pred)
    print("Wynik mape dla ",n,'powtorzen, po dodaniu dodatkowych danych: ', mapeA)

trenowanieA(100)
trenowanieB(100)
trenowanieC(100)
trenowanieD(100)


