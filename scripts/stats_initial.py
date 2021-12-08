import pandas as pd

"""
Scripts pour obtenir les statistiques du fichier dataA.csv
"""

moyenne = []
mediane = []
maximum = []
minimum = []
mode = []
variance = []
valeurs_manquantes = []

data = pd.read_csv('../data/dataA.csv')
colonnes = list(data)
colonnes.remove('DisplayName')

for i in colonnes:

    moyenne.append(data[i].mean())
    maximum.append(data[i].max())
    minimum.append(data[i].min())
    mediane.append(data[i].median())
    mode.append(data[i].mode()[0])
    variance.append(data[i].var(ddof=0))
    valeurs_manquantes.append(data[i].isna().sum())

