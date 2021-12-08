import pandas as pd
import numpy as np
import stats_initial

from sklearn.linear_model import LinearRegression

"""
Scripts pour remplir les valeurs manquantes du fichier dataA.csv et retourne dataB.csv
"""

index_valeurs_manquantes_par_colonne = []
donnees_dataA = pd.read_csv('../data/dataA.csv')
pays = donnees_dataA['DisplayName'].to_numpy()
colonnes = list(donnees_dataA)
colonnes.remove('DisplayName')
nb_rangees = donnees_dataA.shape[0]
index = 0


"""
Applique la médiane de chaque colonne à toutes les cellules vides de chaque colonne
"""
for i in colonnes:
    index_valeurs_manquantes_par_colonne.append(np.where(donnees_dataA[i].isnull())[0])

    donnees_dataA[i].fillna(stats_initial.mediane[index], inplace=True)
    index += 1


def linear_regression(x_train, y_train, x_test):
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    return reg.predict(x_test)


def appliquer_regression(donnees):
    index = 0

    temp2 = donnees.copy()

    for j in colonnes:
        if index_valeurs_manquantes_par_colonne[index].size == 0:
            index += 1
            continue

        temp = donnees.copy()

        x_train = temp.drop(["DisplayName", j], axis=1).drop(index_valeurs_manquantes_par_colonne[index], axis=0)
        x_test = temp.drop(["DisplayName", j], axis=1).loc[temp.index.isin(index_valeurs_manquantes_par_colonne[index])]
        y_train = temp[j].drop(index_valeurs_manquantes_par_colonne[index], axis=0)

        y_test = linear_regression(x_train, y_train, x_test)

        for x in range(len(index_valeurs_manquantes_par_colonne[index])):
            temp2.loc[index_valeurs_manquantes_par_colonne[index][x], j] = y_test[x]

        index += 1

    return temp2


donnees_dataB = appliquer_regression(donnees_dataA)
donnees_dataB = appliquer_regression(donnees_dataB)

donnees_dataB.to_csv("../data/dataB.csv", encoding='utf-8', index=False)


"""
Binarisation des données
"""
donnees_dataC = donnees_dataB.copy()

for n in colonnes:
    mediane = donnees_dataB[n].median()

    for m in range(nb_rangees):
        nouvelle_data = donnees_dataC.loc[m, n]

        if nouvelle_data < mediane:
            donnees_dataC.loc[m, n] = 0

        else:
            donnees_dataC.loc[m, n] = 1


donnees_dataC.to_csv("../data/dataC.csv", encoding='utf-8', index=False)
