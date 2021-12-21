import pandas as pd
import json

donnees = pd.read_csv('../data/dataB.csv')

matrice_correlation = donnees.corr()

"""
Matrice de correlation
"""
json_matrice = json.dumps(matrice_correlation.to_numpy().tolist())
with open("../data/corr.json", "w") as fichier:
    fichier.write(json_matrice)

fichier.close()


"""
Liste de colonnes avec la colonne avec laquelle il y a un max de correlation
"""
with open("../data/corr.json") as fichier:
    matrice = json.load(fichier)

list_max_corr = []
for i in range(len(matrice)):
    max = 0
    indice = -1
    for j in range(len(matrice[i])):
        if i == j:
            continue
        if matrice[i][j] > max:
            max = matrice[i][j]
            indice = j
    list_max_corr.append(indice)

fichier.close()

json_liste_max = json.dumps(list_max_corr)
with open("../data/max_corr.json", "w") as fichier:
    fichier.write(json_liste_max)

fichier.close()

"""
Liste descendante des index de colonnes avec correlation absolue moyenne
"""
matrice_correlation_abs = matrice_correlation.abs().mean()
liste_moy_abs_corr = list(matrice_correlation_abs)
liste_sorted = sorted(range(len(liste_moy_abs_corr)), key=lambda k: liste_moy_abs_corr[k], reverse=True)

json_liste_moy = json.dumps(liste_sorted)
with open("../data/ordre.json", "w") as fichier:
    fichier.write(json_liste_moy)

fichier.close()

