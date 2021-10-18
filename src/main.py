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
from sklearn.metrics import r2_score

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
# Aquests atributs els elimiem per ser dervibables però ho expliquem igualment a l'informe
print(dataset_jordan.isnull().sum())
dataset_jordan= dataset_jordan.replace(np.nan,0)
print(dataset_jordan.isnull().sum())
dataset_lebron= dataset_lebron.replace(np.nan,0)



#minus_plus: pk jordan no es registrava (100% NULL)
#team: els jugadors són diferents
#date: l'edat és el factor que relaciona ambdós jugadors i permet saber date. REDUNDANT

dataset_lebron = dataset_lebron.drop(['minus_plus','team','opp','date','fgp','threep','ftp'],axis=1)
dataset_jordan = dataset_jordan.drop(['minus_plus','team','opp','date','fgp','threep','ftp'],axis=1)


# ----------------------------------------------------------------------------------------------------------------- #
#######Dataset Conversió Atributs String ###########
# ----------------------------------------------------------------------------------------------------------------- #
print(dataset_jordan.head())
# #OPP
# dictOpponents = {}
# def convertOppToId(x):
#     id = len(dictOpponents.values())
#     if x not in dictOpponents:
#         id = id+1
#         dictOpponents[x] = id
#     else:
#         id = dictOpponents[x]
#     return id
#
# dataset_jordan["opp"].replace({x:convertOppToId(x) for x in dataset_jordan["opp"]}, inplace=True)  #canviem str minuts 40:00 a int 40.
# dataset_lebron["opp"].replace({x:convertOppToId(x) for x in dataset_lebron["opp"]}, inplace=True)
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


# ----------------------------------------------------------------------------------------------------------------- #
#!!Mirar amb correlació Pearson
# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
# ----------------------------------------------------------------------------------------------------------------- #
"""
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

# ----------------------------------------------------------------------------------------------------------------- #
#######  Distribució Gausiana de cada Atribut ###########
# ----------------------------------------------------------------------------------------------------------------- #
"""

from scipy.stats import jarque_bera  #3
#from scipy.stats import kstest
#from statsmodels.stats.diagnostic import lilliefors

data_jordan = dataset_jordan.values
dataset_jordan.shape[1]
for i in range(dataset_jordan.shape[1]):

    x = data_jordan[:, i]
    stat, p = jarque_bera(x)
    #stat, p = kstest(x,'norm')
=======
#from scipy.stats import jarque_bera  #3
#from scipy.stats import kstest


data_jordan = dataset_jordan.values
dataset_jordan.shape[1]
for i in range(dataset_jordan.shape[1]):

    x = data_jordan[:, i]
    #stat, p = jarque_bera(x)
    #stat, p = kstest(x,'norm')
    print(p)
    # Interpretación
    alpha = 0.01
    if p > alpha:
        print('Estadisticos=%.3f, p=%.3f' % (stat, p))
        print('La muestra SI parece Gaussiana o Normal (no se rechaza la hipótesis nula H0)'+ dataset_jordan.columns[i])
    else:
        print('La muestra NO parece Gaussiana o Normal(se rechaza la hipótesis nula H0) el atributo '+ dataset_jordan.columns[i])

"""


# ['age','mp','fg','stl','blk','tov']
# ['age', 'result', 'mp', 'fg', 'stl', 'blk', 'tov', 'pts', 'game_score']
# ['result','mp','fg','fga','ft','ft','trb','ast','pts','game_score']
# ----------------------------------------------------------------------------------------------------------------- #
#######  Distribució Gausiana de cada Atribut ###########
# ----------------------------------------------------------------------------------------------------------------- #
"""
=======

# ----------------------------------------------------------------------------------------------------------------- #
#######  Distribució Gausiana de cada Atribut ###########
# ----------------------------------------------------------------------------------------------------------------- #
""""""

plt.figure()
# relacio = sns.pairplot(dataset_jordan[['game','age','result','mp','fg','fga','fgp','three','threeatt','threep','ft','fta','ftp','orb','drb','trb','ast','stl','blk','tov','pts','game_score']])
# relacio = sns.pairplot(dataset_jordan[['game','age','result','mp','fg','fga', 'three', 'threeatt','ft','fta','orb','drb','trb','ast','stl','blk','tov','pts','game_score']])
relacio = sns.pairplot(dataset_jordan[['game','age','result','mp','fg','fga','ft','fta','orb','drb','trb','ast','stl','tov','pts','game_score']])

plt.show()

# plt.figure()
# plt.style.use('dark_background')
# plt.title("Correlació Gausiana Lebron")
# relacio = sns.pairplot(dataset_lebron[['game','age','result','mp','fg','fga','fgp','three','threeatt','threep','ft','fta','ftp','orb','drb','trb','ast','stl','blk','tov','pts','game_score']])
# plt.show()

"""
# ----------------------------------------------------------------------------------------------------------------- #
#*************Generació plot per atribut per veure'n distribució gausiana********
# ----------------------------------------------------------------------------------------------------------------- #
"""
=======

# ----------------------------------------------------------------------------------------------------------------- #
#*************Generació plot per atribut per veure'n distribució gausiana********
# ----------------------------------------------------------------------------------------------------------------- #
""""""


atributs= ['game','age','result','mp','fg','fga','fgp','three','threeatt','threep','ft','fta','ftp','orb','drb','trb','ast','stl','blk','tov','pts','game_score']

for atr in atributs:
    sns.set_theme('notebook', style='dark')
    sns.pairplot(dataset_jordan[[atr]], size=5).fig.suptitle("Correlació Gausiana Jordan: " + atr, y=1)
    plt.show()

    plt.scatter(dataset_jordan['age'], dataset_jordan[atr])
    plt.ylabel(atr);plt.xlabel('age')
    plt.show()


=======


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
# ----------------------------------------------------------------------------------------------------------------- #


###### Estandarització d'atributs  ########
#(x-mitjana)/(max-min)

def estandaritzarMitjana(dataset):
    return (dataset-dataset.mean())/dataset.std()


# dataset_jordan_norm = estandaritzarMitjana(dataset_jordan)
# dataset_lebron_norm = estandaritzarMitjana(dataset_lebron)


# ----------------------------------------------------------------------------------------------------------------- #
# Funcions per la regressió
# ----------------------------------------------------------------------------------------------------------------- #

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

# Aqui abans cel·la 16 jupyter diuen de fer un regressor lineal per cada atribut
# (en nostre cas age,mp, result?) i mirar quin dóna MSE menor

# Quan es treballa en dades n-dimensionals (més d'un atribut), una opció és reduir la seva n-dimensionalitat aplicant
# un Principal Component Analysis (PCA) i quedar-se amb els primers 2 o 3 components, obtenint unes dades que (ara sí)
# poden ser visualitzables en el nou espai. Existeixen altres embeddings de baixa dimensionalitat on poder visualitzar
# les dades?


# data_lebron = dataset_lebron_norm.values
# x_lebron = data_lebron[:, :-1]
# x_lebron = data_lebron[:, :3] #age
# y_lebron = data_lebron[:, -1]
# Dividim dades d'entrenament LEBRON

# data_jordan = dataset_jordan.values
# data_lebron = dataset_lebron.values
# x_jordan = data_jordan[:, :-1]
# y_jordan = data_jordan[:, -1]
#
#
# x_train, y_train, x_val, y_val = split_data(x_lebron, y_lebron)
#
# for i in range(x_train.shape[1]):
#     x_t = x_train[:,i] # seleccionem atribut i en conjunt de train
#     x_v = x_val[:,i] # seleccionem atribut i en conjunt de val.
#     x_t = np.reshape(x_t,(x_t.shape[0],1))
#     x_v = np.reshape(x_v,(x_v.shape[0],1))
#
#     regr = regression(x_t, y_train)
#     error = mse(y_val, regr.predict(x_v)) # calculem error
#     r2 = r2_score(y_val, regr.predict(x_v))
#
#     print("Error en atribut %d: %f" %(data_jordan.columns[i], error))
#


"""
=======
data_lebron=dataset_lebron_norm.values
# x_lebron = data_lebron[:, :-1]
x_lebron = data_lebron[:, :3] #age
y_lebron = data_lebron[:, -1]
# Dividim dades d'entrenament LEBRON

x_train, y_train, x_val, y_val = split_data(x_lebron, y_lebron)



x_t = x_train[:,1] # seleccionem atribut age i del conjunt de train
x_v = x_val[:,1] # seleccionem atribut age i del conjunt de validacio.
x_t = np.reshape(x_t,(x_t.shape[0],1)) # de dataFrame a np per eficiencia
x_v = np.reshape(x_v,(x_v.shape[0],1))

regr = regression(x_t, y_train)
predicted = regr.predict(x_t)
plt.figure()
ax = plt.scatter(x_train[:,1], y_train)
plt.plot(x_t[:,0],predicted,'r')
plt.show()

error = mse(y_val, regr.predict(x_v)) # calculem error
r2 = r2_score(y_val, regr.predict(x_v))

print("Error en atribut %d: %f" %(1, error))
print("R2 score en atribut %d: %f" %(1, r2))

"""

data_jordan = dataset_jordan.values

x_jordan = data_jordan[:, :-1]
y_jordan = data_jordan[:, -1]

dataset_jordan_norm = estandaritzarMitjana(dataset_jordan)
data_jordan_norm = dataset_jordan_norm.values

# x_jordan_norm = data_jordan_norm[:, :-1]
# y_jordan_norm = data_jordan_norm[:, -1]
x_jordan_norm = dataset_jordan_norm[['mp','fg','fga','pts']]
y_jordan_norm = dataset_jordan_norm[['game_score']]

# x_train_norm, y_train_norm, x_test_norm, y_test_norm = split_data(x_jordan_norm.values, y_jordan_norm.values)
from sklearn.model_selection import train_test_split

# train, test = train_test_split(dataset_jordan_norm, test_size=0.2)
x_train_norm,x_test_norm,y_train_norm,y_test_norm = train_test_split(dataset_jordan_norm[['mp','fg','fga','pts']],dataset_jordan_norm[['game_score']],test_size=0.2)

#PCA in action
from sklearn.decomposition import PCA
for i in range(1,5):
    pca = PCA(i)
    x_train_norm_pca = pca.fit_transform(x_train_norm[['mp','fg','fga','pts']].values)
    x_test_norm_pca = pca.transform(x_test_norm[['mp','fg','fga','pts']].values)

    linear_model = LinearRegression()
    linear_model.fit(x_train_norm_pca,y_train_norm)
    preds = linear_model.predict(x_test_norm_pca)

    mse_result = mse(y_test_norm,preds)
    print("PCA: %f - MSE: %f"%(i,mse_result))




=======




z=3