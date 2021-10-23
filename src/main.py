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
selected_Atributes = [['game', 'age', 'result', 'mp', 'fg', 'fga', 'ft', 'fta', 'pts', 'game_score'],
                      ['game', 'age', 'result', 'mp', 'fg', 'fga', 'ft', 'fta', 'orb', 'drb', 'trb', 'ast', 'stl',
                       'tov', 'pts', 'game_score'],
                      ['game', 'age', 'result', 'mp', 'fg', 'fga', 'three', 'threeatt', 'ft', 'fta', 'orb', 'drb',
                       'trb', 'ast', 'stl', 'blk', 'tov', 'pts', 'game_score']]


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
def dimensionalitat(dataset, player_name):
    data = dataset.values
    # separem l'atribut objectiu Y de les caracterísitques X
    x_data = data[:, :-1]
    y_data = data[:, -1]
    print("Dimensionalitat de la BBDD_{}:".format(dataset.shape))
    print("Dimensionalitat de les entrades X_{}: {}".format(player_name, x_data.shape))
    print("Dimensionalitat de l'atribut Y_{}: {}".format(player_name, y_data.shape))


dimensionalitat(dataset_jordan, "jordan")
dimensionalitat(dataset_lebron, "lebron")


#######SELECCIO D'ATRIBUTS###########

#######Datasets Tractament de NaN ###########
# Aquests atributs els elimiem per ser dervibables però ho expliquem igualment a l'informe
def tractar_nulls(dataset, player_name):
    print("Nan del dataset: " + player_name)
    print(dataset.isnull().sum())
    dataset = dataset.replace(np.nan, 0)
    return dataset


dataset_jordan = tractar_nulls(dataset_jordan, "jordan")
dataset_lebron = tractar_nulls(dataset_lebron, "lebron")

# minus_plus: pk jordan no es registrava (100% NULL)
# team,OPP: els jugadors són diferents
# date: l'edat és el factor que relaciona ambdós jugadors i permet saber date. REDUNDANT
# fgp,threep i ftp: són derivables

dataset_lebron = dataset_lebron.drop(['minus_plus', 'team', 'opp', 'date', 'fgp', 'threep', 'ftp'], axis=1)
dataset_jordan = dataset_jordan.drop(['minus_plus', 'team', 'opp', 'date', 'fgp', 'threep', 'ftp'], axis=1)

# ----------------------------------------------------------------------------------------------------------------- #
#######Dataset Conversió Atributs String ###########
# ----------------------------------------------------------------------------------------------------------------- #
print(dataset_jordan.head())


# MP
def minuts_to_int(dataset):
    # canviem str minuts 40:00 a int 40.
    dataset["mp"].replace({x: int(x[:2]) for x in dataset["mp"]}, inplace=True)
    return dataset


# AGE
def age_to_days(dataset):
    dataset["age"].replace({x: int(x[:2]) * 365 + int(x[3:]) for x in dataset["age"]}, inplace=True)
    return dataset


# RESULT
def convert_result(dataset):
    # resultat format: W (+16) agafem el número
    dataset["result"].replace({x: int(x.split('(')[1].split(')')[0]) for x in dataset["result"]}, inplace=True)
    return dataset


# CONVERSIO TOT DATASET
def convert_atributes_type(dataset):
    dataset = minuts_to_int(dataset)
    dataset = age_to_days(dataset)
    return convert_result(dataset)


dataset_jordan = convert_atributes_type(dataset_jordan)
dataset_lebron = convert_atributes_type(dataset_lebron)
print(dataset_jordan.describe())
print(dataset_lebron.describe())


# ----------------------------------------------------------------------------------------------------------------- #
# !!Mirar amb correlació Pearson
# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
# ----------------------------------------------------------------------------------------------------------------- #

def correlacio_pearson(dataset, player_name):
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 10))  # per mida cel·les
    plt.title("Correlació {}".format(player_name))
    sns.heatmap(dataset.corr(), annot=True, linewidths=.5, ax=ax)
    plt.savefig("../figures/pearson_correlation_matrix_" + player_name + ".png")
    plt.show()


#correlacio_pearson(dataset_jordan, "jordan")
#correlacio_pearson(dataset_lebron, "lebron")


# ----------------------------------------------------------------------------------------------------------------- #
#######  Distribució Gausiana de cada Atribut ###########
# ----------------------------------------------------------------------------------------------------------------- #
def testeja_normalitat(dataset, player_name, algoritme):
    from scipy.stats import jarque_bera
    from scipy.stats import chisquare

    data = dataset.values
    print("Resultats normalitat per {}".format(player_name))
    for i in range(dataset.shape[1]):
        x = data[:, i]
        if algoritme == "bera":
            stat, p = jarque_bera(x)
        else:
            stat, p = chisquare(x)  # aqui el Chi Square

        alpha = 0.05
        if p > alpha:
            print('Estadisticos=%.3f, p=%.3f' % (stat, p))
            print(
                'La muestra SI parece Gaussiana o Normal (no se rechaza la hipótesis nula H0)' + dataset.columns[i])
        else:
            print('La muestra NO parece Gaussiana o Normal(se rechaza la hipótesis nula H0) el atributo ' +
                  dataset.columns[i])


# testeja_normalitat(dataset_lebron,"lebron","chi")


# ['age','mp','fg','stl','blk','tov']
# ['age', 'result', 'mp', 'fg', 'stl', 'blk', 'tov', 'pts', 'game_score']
# ['result','mp','fg','fga','ft','ft','trb','ast','pts','game_score']
# ----------------------------------------------------------------------------------------------------------------- #
#######  Distribució Gausiana de cada Atribut ###########
# ----------------------------------------------------------------------------------------------------------------- #
def make_pairplot(dataset, atributs):
    plt.figure()
    sns.pairplot(dataset[atributs])
    plt.show()


# make_pairplot(dataset_jordan, selected_Atributes[0])
# make_pairplot(dataset_lebron, selected_Atributes[0])


# ----------------------------------------------------------------------------------------------------------------- #
# *************Generació plot per atribut per veure'n distribució gausiana********
# ----------------------------------------------------------------------------------------------------------------- #
def make_pairplot_per_atribute(dataset, player_name, atributes):
    for atr in atributes:
        sns.set_theme('notebook', style='dark')
        sns.pairplot(dataset[[atr]], height=5).fig.suptitle("Correlació Gausiana {}: {}".format(player_name, atr), y=1)
        plt.show()

        plt.title("Correlació respecte edat de {} {} ".format(atr, player_name))
        plt.scatter(dataset['age'], dataset[atr])
        plt.ylabel(atr);
        plt.xlabel('age')
        plt.show()


# make_pairplot_per_atribute(dataset_jordan,"jordan",selected_Atributes[2])
# make_pairplot_per_atribute(dataset_lebron,"lebron",selected_Atributes[2])


# Observem que game, age, three, threeatt, threep i stl no tenen distribució gausianan. La resta si.

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
# (x-mitjana)/(max-min)
def estandaritzar_min_max(dataset):
    return (dataset - dataset.min()) / (dataset.max() - dataset.min())


def estandaritzar_mitjana(dataset):
    return (dataset - dataset.mean()) / dataset.std()


# dataset_jordan_norm = estandaritzar_mitjana(dataset_jordan)
# dataset_lebron_norm = estandaritzar_mitjana(dataset_lebron)


###### Estadaritzar + Histogrames + comparar amb histogrames anteriors ########
def make_histogrames(dataset, player_name, atributes):
    dataset_norm = estandaritzar_mitjana(dataset)
    for atr in atributes:
        plt.figure()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Histograma de l'atribut {} de {}".format(atr, player_name))
        ax1.hist(dataset[atr], bins=11, range=[np.min(dataset[atr]), np.max(dataset[atr])], histtype="bar", rwidth=0.8)
        ax1.set(xlabel='Attribute Value', ylabel='Count')
        ax2.hist(dataset_norm[atr], bins=11, range=[np.min(dataset_norm[atr]), np.max(dataset_norm[atr])],
                 histtype="bar", rwidth=0.8)
        ax2.set(xlabel='Normalized value', ylabel='')
        plt.show()


# make_histogrames(dataset_jordan,"jordan",selected_Atributes[0])
# make_histogrames(dataset_lebron,"lebron",selected_Atributes[0])


# ----------------------------------------------------------------------------------------------------------------- #
# Funcions per la regressió
# ----------------------------------------------------------------------------------------------------------------- #

def mse(v1, v2):
    return ((v1 - v2) ** 2).mean()


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
def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0] * train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val


# Aqui abans cel·la 16 jupyter diuen de fer un regressor lineal per cada atribut
# (en nostre cas age,mp, result?) i mirar quin dóna MSE menor


###### MSE i R2 score per Atribut #######################
def error_per_atribut(dataset, player_name, normalize=False):
    if normalize is True:
        dataset_norm = estandaritzar_mitjana(dataset)
        data = dataset_norm.values
    else:
        data = dataset.values
    x_data = data[:, :-1]
    y_data = data[:, -1]
    x_train, y_train, x_val, y_val = split_data(x_data, y_data)

    for i in range(x_train.shape[1]):
        x_t = x_train[:, i]  # seleccionem atribut i el conjunt de train
        x_v = x_val[:, i]  # seleccionem atribut i el conjunt de val.
        x_t = np.reshape(x_t, (x_t.shape[0], 1))
        x_v = np.reshape(x_v, (x_v.shape[0], 1))

        regr = regression(x_t, y_train)
        predicted = regr.predict(x_v)
        plt.figure()
        plt.title("Predicció per {} de {}".format(dataset.columns[i], player_name))
        plt.scatter(x_t, y_train)
        plt.plot(x_v, predicted, 'r')
        plt.show()

        error = mse(y_val, predicted)  # calculem error
        r2 = r2_score(y_val, predicted)

        print("Error en atribut %s: %f" % (dataset.columns[i], error))
        print("R2 score en atribut %s: %f" % (dataset.columns[i], r2))


# error_per_atribut(dataset_jordan,"jordan")
# error_per_atribut(dataset_lebron,"lebron",True)


# Quan es treballa en dades n-dimensionals (més d'un atribut), una opció és reduir la seva n-dimensionalitat aplicant
# un Principal Component Analysis (PCA) i quedar-se amb els primers 2 o 3 components, obtenint unes dades que (ara sí)
# poden ser visualitzables en el nou espai. Existeixen altres embeddings de baixa dimensionalitat on poder visualitzar
# les dades?


# ----------------------------------------------------------------------------------------------------------------- #
# PCA - avaluació dimensionalitat a adequada
# ----------------------------------------------------------------------------------------------------------------- #

def make_pca(dataset, player_name, atributes):
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA

    dataset__norm = estandaritzar_mitjana(dataset[atributes])
    x_norm = dataset__norm[atributes[0:-1]]
    y_norm = dataset__norm[atributes[-1]] # Aquest és l'atribut a predir

    x_train_norm, x_val_norm, y_train_norm, y_val_norm = train_test_split(x_norm, y_norm, test_size=0.2)

    for i in range(1, len(atributes)):
        pca = PCA(i)
        x_train_norm_pca = pca.fit_transform(x_train_norm.values)
        x_test_norm_pca = pca.transform(x_val_norm.values)

        linear_model = LinearRegression()
        linear_model.fit(x_train_norm_pca, y_train_norm)
        preds = linear_model.predict(x_test_norm_pca)

        mse_result = mse(y_val_norm, preds)
        r2 = r2_score(y_val_norm, preds)
        print("PCA %s: %d - MSE: %f - R2: %f" % (player_name, i, mse_result, r2))


make_pca(dataset_jordan, "jordan", selected_Atributes[2])
make_pca(dataset_jordan, "jordan_restricted_40", ['pts', 'fg', 'ft', 'fta', 'fga', 'stl', 'game_score'])
#make_pca(dataset_lebron, "lebron", ['mp', 'fg', 'fga', 'pts'])
#make_pca(dataset_lebron, "lebron", selected_Atributes[2])

z = 3
