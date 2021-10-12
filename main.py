# Agustín Martínez Pérez (1528789)
# Alexandre Moro Rialp (1527046)
# David Sardà Martín (1492054)

import numpy as np
import sklearn as sk
import matplotlib as mp
import scipy as scp
import pandas as pd

# Repo creat
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=",")
    return dataset

dataset_lebron = load_dataset("archive/lebron_career.csv")
dataset_jordan = load_dataset("archive/jordan_career.csv")

dtypes_lebron = dataset_lebron.dtypes
print(dtypes_lebron)
