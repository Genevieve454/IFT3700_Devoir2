import pandas as pd
import json

from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

donnees_B = pd.read_csv('../data/dataB.csv')
donnees_B = donnees_B.drop(columns=['DisplayName'])
colonnes = list(donnees_B)

donnees_C = pd.read_csv('../data/dataC.csv')
donnees_C = donnees_C.drop(columns=['DisplayName'])

"""
Analyse des prédicteurs pour chaque colonne
"""


def linear_regression(x_train, y_train, x_test, y_test):
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    return reg.score(x_test, y_test)


def bayes_classifier(x_train, y_train, x_test, y_test):
    clf = BernoulliNB()
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)


def appliquer_regression_lineaire(donnees):
    liste_score_reg = []
    for i in colonnes:
        liste_temp = []
        X = donnees[i]
        for j in colonnes:
            y = donnees[j]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            X_train = pd.DataFrame(X_train).to_numpy()
            X_test = pd.DataFrame(X_test).to_numpy()
            liste_temp.append(linear_regression(X_train, y_train, X_test, y_test))
        liste_score_reg.append(liste_temp)

    return liste_score_reg


def appliquer_classifieur_bayesien(donnees):
    liste_score_bay = []
    for l in colonnes:
        liste_temp = []
        X = donnees[l]
        for m in colonnes:
            y = donnees[m]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            X_train = pd.DataFrame(X_train).to_numpy()
            X_test = pd.DataFrame(X_test).to_numpy()
            liste_temp.append(bayes_classifier(X_train, y_train, X_test, y_test))
        liste_score_bay.append(liste_temp)

    return liste_score_bay


predicteurs_donneesB = appliquer_regression_lineaire(donnees_B)
predicteurs_donneesC = appliquer_classifieur_bayesien(donnees_C)

"""
Identification des paires de colonnes donnant les meilleurs prédictions
"""


def trouver_meilleures_paires(liste_predicteurs):
    liste_paires = []
    for n in range(len(liste_predicteurs)):
        liste_temp = []
        element_temp = sorted(range(len(liste_predicteurs[n])), key=lambda k: liste_predicteurs[n][k], reverse=True)
        liste_temp.append(element_temp[1])
        liste_temp.append(element_temp[2])
        liste_paires.append(liste_temp)
    return liste_paires


liste_meilleures_col_reg = trouver_meilleures_paires(predicteurs_donneesB)
liste_meilleures_col_bay = trouver_meilleures_paires(predicteurs_donneesC)

json_lineaire_paires = json.dumps(liste_meilleures_col_reg)
with open("../data/lineaire_paires_colonnes.json", "w") as fichier:
    fichier.write(json_lineaire_paires)

fichier.close()

json_bayesien_paires = json.dumps(liste_meilleures_col_bay)
with open("../data/bayesien_paires_colonnes.json", "w") as fichier:
    fichier.write(json_bayesien_paires)

fichier.close()

"""
Trouver les deux meilleurs predicteurs moyen
"""

donnees_B_normalise = donnees_B.copy()

for a in colonnes:
    donnees_B_normalise[a] = (donnees_B[a] - donnees_B[a].mean() / donnees_B[a].std())

predicteurs_donneesB_normalise = appliquer_regression_lineaire(donnees_B_normalise)

liste_moyennes_col_reg = []
for z in range(len(liste_meilleures_col_reg)):
    somme_col1 = 0
    somme_col2 = 0
    for x in range(len(predicteurs_donneesB_normalise)):
        somme_col1 = somme_col1 + predicteurs_donneesB_normalise[x][liste_meilleures_col_reg[z][0]]
        somme_col2 = somme_col2 + predicteurs_donneesB_normalise[x][liste_meilleures_col_reg[z][1]]
    temp = [somme_col1 / len(predicteurs_donneesB_normalise) + somme_col2 / len(predicteurs_donneesB_normalise)]
    liste_moyennes_col_reg.append((somme_col1 / len(predicteurs_donneesB_normalise) + somme_col2 / len(predicteurs_donneesB_normalise))/2)

moyenne_plus_forte_reg = liste_moyennes_col_reg.index(max(liste_moyennes_col_reg))

liste_moyennes_col_bay = []
for o in range(len(liste_meilleures_col_bay)):
    somme_col1 = 0
    somme_col2 = 0
    for p in range(len(predicteurs_donneesC)):
        somme_col1 = somme_col1 + predicteurs_donneesC[p][liste_meilleures_col_bay[o][0]]
        somme_col2 = somme_col2 + predicteurs_donneesC[p][liste_meilleures_col_bay[o][1]]
    temp = [somme_col1 / len(predicteurs_donneesC) + somme_col2 / len(predicteurs_donneesC)]
    liste_moyennes_col_bay.append((somme_col1 / len(predicteurs_donneesB_normalise) + somme_col2 / len(predicteurs_donneesC))/2)

moyenne_plus_forte_bay = liste_moyennes_col_bay.index(max(liste_moyennes_col_bay))

meilleurs_predicteurs_reg = liste_meilleures_col_reg[moyenne_plus_forte_reg]
meilleurs_predicteurs_bay = liste_meilleures_col_bay[moyenne_plus_forte_bay]

