# Agustín Martínez Pérez (1528789)
# Alexandre Moro Rialp (1527046)
# David Sardà Martín (1492054)

import numpy as np
import sklearn as sk
import matplotlib as mp
import scipy as scp
import pandas as pd
# %matplotlib notebook
from matplotlib import pyplot as plt

# Repo creat
# Push de prova

def load_dataset(path):
    return pd.read_csv(path, header=0, delimiter=",")

#######Datasets atributs types###########
dataset_jordan = load_dataset("/home/alexandre/Desktop/APC/Plab/practica1-regressio-GPA302-1030/archive/jordan_career.csv")
dataset_lebron = load_dataset("/home/alexandre/Desktop/APC/Plab/practica1-regressio-GPA302-1030/archive/lebron_career.csv")

print(dataset_lebron.dtypes)

#######Datasets Dimensionalitat###########
data_jordan = dataset_jordan.values
data_lebron = dataset_lebron.values

x_jordan = data_jordan[:, :2]
y_jordan = data_jordan[:, 2]
print("Dimensionalitat de la BBDD_jordan:", dataset_jordan.shape)
print("Dimensionalitat de les entrades X_jordan", x_jordan.shape)
print("Dimensionalitat de l'atribut Y_jordan", y_jordan.shape)

x_lebron = data_lebron[:, :2]
y_lebron = data_lebron[:, 2]
print("Dimensionalitat de la BBDD_lebron:", dataset_lebron.shape)
print("Dimensionalitat de les entrades X_lebron", x_lebron.shape)
print("Dimensionalitat de l'atribut Y_lebron", y_lebron.shape)

#######Selecció d'Atributs###########
#!!Mirar amb correlació
#minus_plus: pk jordan no es registrava (100% NULL)
#team i opp: els jugadors són diferents
#result,date

dataset_lebron = dataset_lebron.drop(['minus_plus','team','opp','date','result'],axis=1)
dataset_jordan = dataset_jordan.drop(['minus_plus','team','opp','date','result'],axis=1)

#######Datasets Tractament de NaN ###########
print(dataset_jordan.isnull().sum())
dataset_jordan= dataset_jordan.replace(np.nan,0)
print(dataset_jordan.isnull().sum())

dataset_lebron= dataset_lebron.replace(np.nan,0)

#######Dataset Conversió Atributs String ###########
print(dataset_jordan.head())
##MP
dataset_jordan["mp"].replace({x:int(x[:2]) for x in dataset_jordan["mp"]}, inplace=True)  #canviem str minuts 40:00 a int 40.
dataset_lebron["mp"].replace({x:int(x[:2]) for x in dataset_lebron["mp"]}, inplace=True)
#AGE
dataset_jordan["age"].replace({x:int(x[:2])+int(x[3:])/365 for x in dataset_jordan["age"]}, inplace=True)
dataset_lebron["age"].replace({x:int(x[:2])+int(x[3:])/365 for x in dataset_lebron["age"]}, inplace=True)

print(dataset_jordan.describe())
print(dataset_lebron.describe())

#######  Distribució Gausiana de cada Atribut ###########
plt.figure()
data_jordan = dataset_jordan.values
x_jordan = data_jordan[:, :2]
y_jordan = data_jordan[:, :2]
ax = plt.scatter(x_jordan[:,0], y_jordan)
plt.show()
b=3