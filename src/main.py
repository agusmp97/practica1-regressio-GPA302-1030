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
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import jarque_bera
from scipy.stats import chisquare
import random

# Variables globals
# selected_attributes = [['game', 'age', 'result', 'mp', 'fg', 'fga', 'ft', 'fta', 'pts', 'game_score'],
#                      ['game', 'age', 'result', 'mp', 'fg', 'fga', 'ft', 'fta', 'orb', 'drb', 'trb', 'ast', 'stl',
#                       'tov', 'pts', 'game_score'],
#                      ['game', 'age', 'result', 'mp', 'fg', 'fga', 'three', 'threeatt', 'ft', 'fta', 'orb', 'drb',
#                       'trb', 'ast', 'stl', 'blk', 'tov', 'pts', 'game_score']]

# Etiquetes que identifiquen les característiques del dataset per utilitzar-les com a títols dels plots.
x_labels = [
    'Posició partit temporada',
    'Edat en dies',
    'Diff. de punts',
    'Minuts jugats/partit',
    'Llançaments anotats',
    'Llançaments intentats',
    'Triples anotats',
    'Triples intentats',
    'Tirs lliures anotats',
    'Tirs lliures intentats',
    'Rebots en atac',
    'Rebots en defensa',
    'Rebots jugats',
    'Assistències',
    'Pilotes robades',
    'Taps',
    'Contraatacs',
    'Punts'
]


# Funció que carrega el dataset des del fitxer especificat per paràmetre.
# Retorna un DataFrame.
def load_dataset(path):
    return pd.read_csv(path, header=0, delimiter=",")


# DataFrames que contenen les dades dels dos jugadors a comparar.
dataset_jordan = load_dataset("../data/archive/jordan_career.csv")
dataset_lebron = load_dataset("../data/archive/lebron_career.csv")

# +--------------------------+
# | VISUALITZACIÓ INFORMACIÓ |
# +--------------------------+

# Mostra els primers 5 registres dels DataFrames dels jugadors
print("HEAD del dataset de Jordan")
print(dataset_jordan.head())
print("------------------------------------")
print("HEAD del dataset de LeBron")
print(dataset_lebron.head())
print("------------------------------------")


# Funció que mostra per consola els tipus de dades de les característiques dels DataFrames dels jugadors.
def print_data_types():
    print("------------------------------------")
    print("Tipus de dades: dataset LeBron James")
    print(dataset_lebron.dtypes)
    print("------------------------------------")
    print("Tipus de dades: dataset Michael Jordan")
    print(dataset_jordan.dtypes)
    print("------------------------------------")


print_data_types()


# Funció que mostra la dimensionalitat dels DataFrames
def df_dimensionality(dataset, player_name):
    data = dataset.values
    # separem l'atribut objectiu Y de les caracterísitques X
    x_data = data[:, :-1]  # Característiques del jugador
    y_data = data[:, -1]  # Variable objectiu (target)
    print("Dimensionalitat del DataFrame de {}: {}:".format(player_name, dataset.shape))
    print("Dimensionalitat de les característiques (X) de {}: {}".format(player_name, x_data.shape))
    print("Dimensionalitat de la variable objectiu (Y) de {}: {}".format(player_name, y_data.shape))
    print("------------------------------------")


df_dimensionality(dataset_jordan, "Jordan")
df_dimensionality(dataset_lebron, "LeBron")


# +-----------------------+
# | TRACTAMENT D'ATRIBUTS |
# +-----------------------+

# Funció que substitueix els valors nuls dels datasets dels jugadors pel valor numèric '0'.
def nan_treatment(dataset, player_name):
    print("Eliminació 'NaN' del DataFrame {}: ".format(player_name))
    print(dataset.isnull().sum())
    print("------------------------------------")
    dataset = dataset.replace(np.nan, 0)
    return dataset


dataset_jordan = nan_treatment(dataset_jordan, "Jordan")
dataset_lebron = nan_treatment(dataset_lebron, "LeBron")

# Eliminació d'atributs no necessaris dels DataFrames
dataset_lebron = dataset_lebron.drop(['minus_plus', 'team', 'opp', 'date', 'fgp', 'threep', 'ftp'], axis=1)
dataset_jordan = dataset_jordan.drop(['minus_plus', 'team', 'opp', 'date', 'fgp', 'threep', 'ftp'], axis=1)


# Funció que modifica els 'Minutes Played' (mp) en format 'min:seg' a 'min' (tipus 'int')
def minutes_to_int(dataset):
    dataset["mp"].replace({x: int(x[:2]) for x in dataset["mp"]}, inplace=True)
    return dataset


# Funció que modifica l'edat (age) en format 'anys+dies' a 'dies' (tipus 'int')
def age_to_days(dataset):
    dataset["age"].replace({x: int(x[:2]) * 365 + int(x[3:]) for x in dataset["age"]}, inplace=True)
    return dataset


# Funció que modifica el resultat (result) en format 'W/L (+/-diff)' a '+/-diff' (tipus 'int')
def convert_result(dataset):
    dataset["result"].replace({x: int(x.split('(')[1].split(')')[0]) for x in dataset["result"]}, inplace=True)
    return dataset


# Funció que crida als canvis de formats dels atributs necessaris.
# Retorna un DataFrame amb els resultats convertits als formats correctes.
def reformat_attributes(dataset):
    dataset = minutes_to_int(dataset)
    dataset = age_to_days(dataset)
    return convert_result(dataset)


dataset_jordan = reformat_attributes(dataset_jordan)
dataset_lebron = reformat_attributes(dataset_lebron)

print("Estadístiques DataFrame Jordan:")
print(dataset_jordan.describe())
print("------------------------------------")
print("Estadístiques DataFrame LeBron:")
print(dataset_lebron.describe())
print("------------------------------------")


# +-----------------------+
# | CORRELACIÓ D'ATRIBUTS |
# +-----------------------+
# Funció que genera la matriu de correlació de Pearson d'un DataFrame i genera el plot
def pearson_correlation(dataset, player_name):
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 10))  # figsize controla l'amplada i alçada de les cel·les de la matriu
    plt.title("Matriu de correlació de Pearson: DataFrame {}".format(player_name))
    sns.heatmap(dataset.corr(), annot=True, linewidths=.5, ax=ax)
    plt.savefig("../figures/pearson_correlation_matrix_" + player_name + ".png")
    plt.show()


# pearson_correlation(dataset_jordan, "Jordan")
# pearson_correlation(dataset_lebron, "LeBron")


# Funció que determina si la distribució de les dades és Gaussiana o no.
# Permet seleccionar entre els algoritmes Jarque-Bera i Chisquare.
def normality_test(dataset, player_name, algoritme):
    data = dataset.values
    print("Resultats del test de normalitat pel DataFrame {}:".format(player_name))
    for i in range(dataset.shape[1]):
        x = data[:, i]
        if algoritme == "bera":
            stat, p = jarque_bera(x)
        else:
            stat, p = chisquare(x)

        alpha = 0.05
        if p > alpha:
            print('Estadístiques=%.3f, p=%.3f' % (stat, p))
            print("{}: SI és Gaussiana la distribució.".format(dataset.columns[i]))
        else:
            print("{}: NO és Gaussiana la distribució.".format(dataset.columns[i]))

    print("------------------------------------")


# normality_test(dataset_lebron, "LeBron", "chi")


# Funció que crea representa la relació entre cada parella de característiques (pairplot) del DataSet.
def make_pairplot(dataset, atributs):
    plt.figure()
    sns.pairplot(dataset[atributs])
    plt.show()


# pairplot_attr = ['game', 'age', 'result', 'mp', 'fg', 'fga', 'ft', 'fta', 'pts', 'game_score']
# make_pairplot(dataset_jordan, pairplot_attr)
# make_pairplot(dataset_lebron, pairplot_attr)


# Funció que genera els plots per comparar les diferents característiques amb ella mateixa i amb l'atribut 'age'.
def make_pairplot_per_atribute(dataset, player_name, attributes):
    for attr in attributes:
        sns.set_theme('notebook', style='dark')
        sns.pairplot(dataset[[attr]], height=5).fig.suptitle("Correlació Gausiana {}: {}".format(player_name, attr),
                                                             y=1)
        plt.show()

        plt.title("Correlació respecte edat de {} {} ".format(attr, player_name))
        plt.scatter(dataset['age'], dataset[attr])
        plt.ylabel(attr)
        plt.xlabel('age')
        plt.show()


# pairplot_attr = ['game', 'age', 'result', 'mp', 'fg', 'fga', 'three', 'threeatt', 'ft', 'fta', 'orb', 'drb', 'trb',
#                 'ast', 'stl', 'blk', 'tov', 'pts', 'game_score']
# make_pairplot_per_atribute(dataset_jordan, "Jordan", pairplot_attr)
# make_pairplot_per_atribute(dataset_lebron, "LeBron", pairplot_attr)


# +------------------+
# | REGRESSIÓ LINEAL |
# +------------------+

# Funció que estandarditza els valors del DataFrame del jugador, per tal de permetre fer que les diferents
# característiques siguin comparables entre elles.
def standardize_mean(dataset):
    return (dataset - dataset.mean(0)) / dataset.std(0)


dataset_jordan_norm = standardize_mean(dataset_jordan)
dataset_lebron_norm = standardize_mean(dataset_lebron)


# Funció que crea els histogrames de les característiques sense normalitzar i normalitzades, el que permet fer la
# comparació entre els conjunts de dades.
def make_histograms(dataset, player_name, attributes):
    dataset_norm = standardize_mean(dataset)
    for attr in attributes:
        plt.figure()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Histograma de l'atribut {} de {}".format(attr, player_name))
        ax1.hist(dataset[attr], bins=11, range=[np.min(dataset[attr]), np.max(dataset[attr])], histtype="bar",
                 rwidth=0.8)
        ax1.set(xlabel='Attribute Value', ylabel='Count')
        ax2.hist(dataset_norm[attr], bins=11, range=[np.min(dataset_norm[attr]), np.max(dataset_norm[attr])],
                 histtype="bar", rwidth=0.8)
        ax2.set(xlabel='Normalized value', ylabel='')
        plt.show()


# hist_attr = ['game', 'age', 'result', 'mp', 'fg', 'fga', 'ft', 'fta', 'pts', 'game_score']
# make_histograms(dataset_jordan, "Jordan", hist_attr)
# make_histograms(dataset_lebron, "LeBron", hist_attr)

# Funció que calcula la mitjana de l'error quadràtic que es comet entre el valor real i la predicció feta pel model.
# v1 és el valor real de les dades del DataSet.
# v2 és el valor predit pel model.
def mse(v1, v2):
    return ((v1 - v2) ** 2).mean()


# Funció que implementa la regressió i entrena el model.
# x és el conjunt de característiques d'entrada.
# y és el valor objectiu, al qual el model s'ha d'apropar al màxim en la seva predicció.
def regression(x, y):
    regr = LinearRegression()  # Crea un objecte de regressió lineal amb sklearn.
    regr.fit(x, y)  # Entrena el model per a que, a partir de les entrades (x), pugui predir el resultat (y)
    return regr  # Retorna el model entrenat


# Funció que estandarditza els valors de les característiques d'entrenament del model.
def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t


# Funció que separa les dades del dataset per tenir un 80% d'entrenament i un 20% de validació.
# Retorna els conjunts d'entrenament x_train, y_train i els conjunts de validació x_val i y_val.
def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[
                            0])  # Genera un array de la mida del nombre de característiques on, a cada posició, posa l'index de la posició
    np.random.shuffle(indices)  # Intercanvia els valors, aleatòriament, del array d'índexs.
    n_train = int(np.floor(
        x.shape[0] * train_ratio))  # Calcula la posició a partir de la qual agafar els registres de x per a entrenament
    indices_train = indices[:n_train]  # Selecciona els índexs a partir de la posició n_train, que seran els de training
    indices_val = indices[n_train:]  # Selecciona els índexs fins a la posició n_train, que seran els de validació
    x_train = x[indices_train, :]  # Selecciona els atributs d'entrada d'entrenament
    y_train = y[indices_train]  # Selecciona els atributs objectiu d'entrenament
    x_val = x[indices_val, :]  # Selecciona els atributs d'entrada de validació
    y_val = y[indices_val]  # Selecciona els atributs objectiu de validació
    return x_train, y_train, x_val, y_val


# Funció que fa l'entrenament del model amb el regressor univariable, sobre les característiques del dataset
# passat per paràmetre.
def error_per_atribut(dataset, player_name, normalize=False):
    if normalize is True:
        dataset_norm = standarize(dataset)  # Estandarditza els valors de les característiques del dataset.
        data = dataset_norm.values
    else:
        data = dataset.values

    x_data = data[:, :-1]
    y_data = data[:, -1]

    # Separa les dades en dades d'entrenament i de validació
    x_train, y_train, x_val, y_val = split_data(x_data, y_data)

    for i in range(x_train.shape[1]):
        x_t = x_train[:, i]  # Selecciona atribut i el conjunt de training.
        x_v = x_val[:, i]  # Selecciona atribut i el conjunt de validació.
        x_t = np.reshape(x_t, (x_t.shape[0], 1))
        x_v = np.reshape(x_v, (x_v.shape[0], 1))

        regr = regression(x_t, y_train)  # Retorna el model entrenat
        predicted = regr.predict(x_v)  # Fa la predicció, amb les dades de validació, sobre el model entrenat

        error = str(round(mse(y_val, predicted), 3))
        r2 = str(round(r2_score(y_val, predicted), 3))

        # Generació del plot
        plt.figure()
        plt.title("Predicció: {}  -   MSE: {}   R2: {}".format((dataset.columns[i]).upper(), error, r2))
        plt.xlabel(x_labels[i])
        plt.ylabel('Game_score')
        plt.scatter(x_v, y_val)
        plt.plot(x_v, predicted, 'r')
        # plt.savefig("../../figures/predic_univ_std_{}_{}.png".format(dataset.columns[i], player_name))
        plt.show()

        print("MSE en atribut %s: %s" % (dataset.columns[i], error))
        print("R2 score en atribut %s: %s" % (dataset.columns[i], r2))


error_per_atribut(dataset_jordan, "Jordan")


# Descomentar si es vol entrenar el model amb les dades normalitzades.
# error_per_atribut(dataset_jordan, "Jordan", True)


# +----------------------------------------+
# | PCA - transformació de dimensionalitat |
# +----------------------------------------+

# Funció que genera el regressor multivariable entrenat amb un dataset al qual se li ha aplicat una transformació
# de dimensionalitat (amb el mètode PCA).
def make_pca(dataset, player_name, atributes, print_plot):
    dataset__norm = standardize_mean(dataset[atributes])
    x_norm = dataset__norm[atributes[0:-1]]
    y_norm = dataset__norm[atributes[-1]]  # Aquest és l'atribut a predir

    # Separa les dades entre el conjunt d'entrenament i de validació
    x_train_norm, x_val_norm, y_train_norm, y_val_norm = train_test_split(x_norm, y_norm, test_size=0.2)

    # Vectors que emmagatzemaran les dades per generar els gràfics de l'evolució de l'error.
    mse_vect = []
    i_vect = []
    r2_vect = []

    # Fa la transformació de dimensionalitat per un nombre incremental de components principals.
    for i in range(1, len(atributes)):
        pca = PCA(i)
        x_train_norm_pca = pca.fit_transform(x_train_norm.values)  # Transformació de les dades de training.
        x_test_norm_pca = pca.transform(x_val_norm.values)  # Transformació de les dades de validació.

        total_var = pca.explained_variance_ratio_.sum() * 100  # Variança total
        lab = {str(j): f"PC {j + 1}" for j in range(i)}
        lab['color'] = 'Game_score'

        fig = px.scatter_matrix(
            x_test_norm_pca,
            color=y_val_norm,
            dimensions=range(i),
            labels=lab,
            title=f'Total Explained Variance: {total_var:.2f}%',
        )
        fig.update_traces(diagonal_visible=False)
        fig.show()

        linear_model = LinearRegression()  # Crea el model regressor
        linear_model.fit(x_train_norm_pca, y_train_norm)  # Entrena el regressor amb les dades d'entrenament
        preds = linear_model.predict(x_test_norm_pca)  # Fa la predicció sobre les dades de validació

        mse_result = mse(y_val_norm, preds)
        i_vect.append(i)
        mse_vect.append(mse_result)

        r2 = r2_score(y_val_norm, preds)
        r2_vect.append(r2)
        print("PCA %s: %d - MSE: %f - R2: %f" % (player_name, i, mse_result, r2))

    if print_plot:
        plt.figure()
        ax = plt.scatter(x_test_norm_pca[:, 0], y_val_norm)
        plt.plot(x_test_norm_pca[:, 0], preds, 'r')
        plt.show()

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(i_vect, r2_vect, 'b', label='R2_Score')
    ax.plot(i_vect, mse_vect, 'r', label='MSE')
    ax.legend(bbox_to_anchor=(1, 0.8))
    plt.title("Error per dimensionalitat PCA")
    plt.show()


# pca_attr = ['game', 'age', 'result', 'mp', 'fg', 'fga', 'three', 'threeatt', 'ft', 'fta', 'orb', 'drb',
#            'trb', 'ast', 'stl', 'blk', 'tov', 'pts', 'game_score']
# make_pca(dataset_jordan, "jordan_Dataset", pca_attr, False)

# pca_attr = ['pts', 'fg', 'ft', 'fta', 'fga', 'stl', 'game_score']
# make_pca(dataset_jordan, "jordan_Pearson_40", pca_attr, True)

# pca_attr = ['pts','ast','drb','fg', 'ft', 'fta', 'tov', 'trb', 'game_score']
# make_pca(dataset_jordan, "jordan_Gaussian", pca_attr, True)


# Aquesta classe representa la implementació de l'algoritme de Descens de Gradient.
class Regressor(object):
    def __init__(self, w0, w1, alpha, x_t, y_t, x_val, y_val):
        self.w0 = w0  # Inicialització dels pesos inicials
        self.w1 = w1  # Inicialització dels pesos inicials
        self.alpha = alpha  # Coeficient de modificació dels pesos
        self.x = x_t  # Conjunt de dades d'entrada d'entrenament
        self.y = y_t  # Conjunt de dades objectiu d'entrenament
        self.x_val = x_val  # Conjunt de dades d'entrada de validació
        self.y_val = y_val  # Conjunt de dades objectiu de validació

    # Aquesta funció fa la predicció dels valors objectiu a partir dels pesos actuals.
    def predict(self, x):
        hy = []
        for i in range(len(x)):
            hy.append(self.w0 + self.w1 * x[i])

        return hy

    # Aquesta funció actualitza els pesos de les característiques amb els nous valors en funció del signe de la
    # derivada (pendent de la funció de cost).
    def __update(self, hy, y):
        # Variables auxiliars per emmagatzemar els sumatoris de la diferència entre la predicció i el valor real.
        z = []  # Per w0
        z2 = []  # Per w1
        for i in range(len(hy)):
            z.append((hy[i] - y[i]))
            z2.append((hy[i] - y[i]) * self.x[i])

        # Actualització dels pesos.
        self.w0 = self.w0 - self.alpha * (1 / len(hy) * sum(z))
        self.w1 = self.w1 - self.alpha * (1 / len(hy) * sum(z2))

    # Aquesta funció realitza l'entrenament del model.
    def train(self, max_iter, epsilon):
        iteracions = 1
        mse_anterior = 100
        mse_actual = 100
        millora = 100
        predides = 0
        i_vect = []
        error = []

        # Bucle que s'executa fins a esgotar les iteracions o convergir.
        while iteracions <= max_iter and epsilon < millora:
            # Per anar reduïnt l'increment cada certes iteracions, per tal de no sortir dels valors mínims trobats.
            if iteracions % 200 == 0:
                self.alpha = self.alpha / 10
            if iteracions != 1:  # Actualitza els pesos a partir de la segona iteració.
                self.__update(predides, self.y)

            predides = self.predict(self.x)  # Fa la predicció sobre el conjunt de dades d'entrenament.
            mse_actual = mse(predides, self.y)  # Calcula el MSE segons la predicció feta.
            r2_r = r2_score(predides, self.y)  # Calcula la correlació entre la predicció i el valor real.
            millora = abs(mse_anterior - mse_actual)  # Calcula la millora de l'error respecte la iteració anterior.
            i_vect.append(iteracions)  # Afegeix el número d'iteració a un vector per relacionar-lo amb el MSE
            error.append(mse_actual)  # Serveix per relacionar el MSE amb la iteració
            mse_anterior = mse_actual  # Actualitza el MSE
            iteracions += 1

        y_pred = self.predict(self.x_val)  # Fa la predicció sobre les dades de validació amb el model entrenat
        print(r2_r)

        return iteracions, self.w0, self.w1, mse_actual, r2_r, i_vect, error, y_pred

# Funció que inicialitza de forma aleatòria els valors inicials dels pesos del model.
def Montecarlo(x_train, x_val, y_train, y_val):
    mse_actual_min = 999999

    for i in range(20):  # Nombre de descensos de gradient aplicats
        # Inicialització aleatòria
        w0 = random.gauss(0.5, 0.5)
        w1 = random.gauss(0.5, 1)

        reg = Regressor(w0, w1, 0.00005, x_train.values, y_train.values, x_val.values, y_val.values)
        # Entrena el model
        iteracions, w0, w1, mse_actual, r2_r, i_vect, error, pred = reg.train(1000000, 0.000000001)

        if mse_actual < mse_actual_min:  # Emmagatzema els paràmetres del model amb millors resultats
            iteracions_min, w0_min, w1_min, mse_actual_min, r2_r_min, i_vect_min, error_min, pred_min = \
                iteracions, w0, w1, mse_actual, r2_r, i_vect, error, pred

    plt.figure()
    plt.title("Predicció D. Gradient MSE: {}   R2: {}".format(round(mse_actual_min, 3), round(r2_r_min, 3)))
    plt.xlabel('pts')
    plt.ylabel('Game_score')
    plt.scatter(x_val.values, y_val.values)
    plt.plot(x_val.values, pred_min, 'r')
    # plt.savefig("../../figures/predic_univ_std_{}_{}.png".format(dataset.columns[i], player_name))
    plt.show()

    return iteracions_min, w0_min, w1_min, mse_actual_min, r2_r_min, pred_min


x_train, x_val, y_train, y_val = train_test_split(standardize_mean(dataset_jordan['pts']),
                                                  standardize_mean(dataset_jordan[["game_score"]]),
                                                  test_size=0.2)
print(Montecarlo(x_train, x_val, y_train, y_val))