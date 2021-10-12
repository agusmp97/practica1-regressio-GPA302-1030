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
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import normaltest

# Repo creat
# ----------------------------------------------------------------------------------------------------------------- #
# Primera part C EDA
# ----------------------------------------------------------------------------------------------------------------- #
def load_dataset(path):
    return pd.read_csv(path, header=0, delimiter=",")

#######Datasets atributs types###########
dataset_jordan = load_dataset("../data/archive/jordan_career.csv")
dataset_lebron = load_dataset("../data/archive/lebron_career.csv")

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

#######Datasets Tractament de NaN ###########
print(dataset_jordan.isnull().sum())
dataset_jordan= dataset_jordan.replace(np.nan,0)
print(dataset_jordan.isnull().sum())
dataset_lebron= dataset_lebron.replace(np.nan,0)



#minus_plus: pk jordan no es registrava (100% NULL)
#team: els jugadors són diferents
#date: l'edat és el factor que relaciona ambdós jugadors i permet saber date. REDUNDANT

dataset_lebron = dataset_lebron.drop(['minus_plus','team','date'],axis=1)
dataset_jordan = dataset_jordan.drop(['minus_plus','team','date'],axis=1)



#######Dataset Conversió Atributs String ###########
print(dataset_jordan.head())
#OPP
dictOpponents = {}
def convertOppToId(x):
    id = len(dictOpponents.values())
    if x not in dictOpponents:
        id = id+1
        dictOpponents[x] = id
    else:
        id = dictOpponents[x]
    return id

dataset_jordan["opp"].replace({x:convertOppToId(x) for x in dataset_jordan["opp"]}, inplace=True)  #canviem str minuts 40:00 a int 40.
dataset_lebron["opp"].replace({x:convertOppToId(x) for x in dataset_lebron["opp"]}, inplace=True)
#MP
dataset_jordan["mp"].replace({x:int(x[:2]) for x in dataset_jordan["mp"]}, inplace=True)  #canviem str minuts 40:00 a int 40.
dataset_lebron["mp"].replace({x:int(x[:2]) for x in dataset_lebron["mp"]}, inplace=True)
#AGE
dataset_jordan["age"].replace({x:int(x[:2])+int(x[3:])/365 for x in dataset_jordan["age"]}, inplace=True)
dataset_lebron["age"].replace({x:int(x[:2])+int(x[3:])/365 for x in dataset_lebron["age"]}, inplace=True)
#RESULT
def convertResult(x):
    aux = x.split('(')
    aux = aux[1].split(')')[0]
    return int(aux)
dataset_jordan["result"].replace({x:convertResult(x) for x in dataset_jordan["result"]}, inplace=True)
dataset_lebron["result"].replace({x:convertResult(x) for x in dataset_lebron["result"]}, inplace=True)

print(dataset_jordan.describe())
print(dataset_lebron.describe())

"""
#!!Mirar amb correlació Pearson
# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
plt.figure()
fig, ax = plt.subplots(figsize=(20,10)) #per mida cel·les
plt.title("Correlació Jordan")
axu1 = sns.heatmap(dataset_jordan.corr(), annot=True, linewidths=.5,ax=ax)
plt.show()
plt.figure()
fig2, ax2 = plt.subplots(figsize=(20,10))
plt.title("Correlació Lebron")
aux2 = sns.heatmap(dataset_lebron.corr(), annot=True, linewidths=.5,ax=ax2)
plt.show()
"""
"""
#######  Distribució Gausiana de cada Atribut ###########
for i in range(21):
    #plt.figure()
    #data_jordan = dataset_jordan.values
    #x_jordan = data_jordan[:, :i]
    #y_jordan = data_jordan[:, i]
    #ax = plt.scatter(x_jordan[: ,0], y_jordan)
    #plt.show()
    plt.figure()
    plt.hist(y_jordan )
    plt.title('Histograma de una variable')
    plt.xlabel('Valor de la variable')
    plt.ylabel('Conteo')
    plt.show()
    stat, p = normaltest(y_jordan)

    # Interpretación
    alpha = 0.05
    if p > alpha:
        print('Estadisticos=%.3f, p=%.3f' % (stat, p))
        print('La muestra SI parece Gaussiana o Normal (no se rechaza la hipótesis nula H0)'+ dataset_jordan.columns[i])
    else:
        print('La muestra NO parece Gaussiana o Normal(se rechaza la hipótesis nula H0) el atributo '+ dataset_jordan.columns[i])
"""


"""
#######  Distribució Gausiana de cada Atribut ###########
plt.figure()
plt.title("Correlació Gausiana Jordan")
relacio = sns.pairplot(dataset_jordan[['game','age','result','mp','fg','fga','fgp','three','threeatt','threep','ft','fta','ftp','orb','drb','trb','ast','stl','blk','tov','pts','game_score']])
#relacio = sns.pairplot(dataset_jordan[['game','age','result','mp','fg','fga','fgp',]])

plt.show()

plt.figure()
plt.style.use('dark_background')
plt.title("Correlació Gausiana Lebron")
relacio = sns.pairplot(dataset_lebron[['game','age','result','mp','fg','fga','fgp','three','threeatt','threep','ft','fta','ftp','orb','drb','trb','ast','stl','blk','tov','pts','game_score']])
plt.show()

#*************Generació plot per atribut per veure'n distribució gausiana********
atributs= ['game','age','result','mp','fg','fga','fgp','three','threeatt','threep','ft','fta','ftp','orb','drb','trb','ast','stl','blk','tov','pts','game_score']

for atr in atributs:
    sns.set_theme('notebook', style='dark')
    sns.pairplot(dataset_jordan[[atr]], size=5).fig.suptitle("Correlació Gausiana Jordan: " + atr, y=1)
    plt.show()

    plt.scatter(dataset_jordan['age'], dataset_jordan[atr])
    plt.ylabel(atr);plt.xlabel('age')
    plt.show()
"""



z=3
#Observem que game, age, three, threeatt, threep i stl no tenen distribució gausianan. La resta si.

# L'atribut objectiu (sortida Y del model) serà game_score perquè és l'element representatiu del rendiment del jugador
# en un partit i el nostre objectiu és poder predir quan lebron podrà atrapar a jordan en el futur pel què fa a
# rendiment global de la seva carrera.
# Pel què fa  a les entrades o caracterísitques del model, considerem que els atributs que hem d'agafar són aquells que
# no són directament dependents del rendiment del jugador, és a dir, l'edat, els minuts, l'opponent i el resultat del
# partit. Això també és així perquè, de fet, el càlcul del game_score es fa seguint una fórmula que mira els atributs de
# de les accions de ljugador com els punts anotats, els intents, les recuperacions i els rebots, etc. Si el nostre model
# depengués d'aquestes entrades aleshores no aportaria cap predició sinó que simplement faria la constatació del càlcul.


# ----------------------------------------------------------------------------------------------------------------- #
# Part regressió lineal



###### Estandarització d'atributs  ########
#(x-mitjana)/(max-min)

def estandaritzarMitjana(dataset):
    return (dataset-dataset.mean())/dataset.std()


dataset_jordan_norm = estandaritzarMitjana(dataset_jordan)
dataset_lebron_norm = estandaritzarMitjana(dataset_lebron)

# Funcions per la regressió
def mse(v1, v2):
    return ((v1 - v2)**2).mean()

def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()
    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)
    # Retornem el model entrenat
    return regr

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

# Dividim dades en 80% train i 20½ Cvalidation

# Estadaritzar + Histogrames + comparar amb histogrames anteriors


def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val

# Dividim dades d'entrenament
#x_train, y_train, x_val, y_val = split_data(x, y)