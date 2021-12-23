import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

donnees_B = pd.read_csv('../data/dataB.csv')
colonnes = list(donnees_B)

X = donnees_B.iloc[:, 1:len(donnees_B)]
y = donnees_B.iloc[:, 0]

X = StandardScaler().fit_transform(X)

def pca_en_2D(donnees):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(donnees)

    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['Componant 1', 'Componant 2'])

    finalDf = pd.concat([principalDf, donnees_B['DisplayName']], axis=1)

    fig = plt.figure(figsize=(32, 32))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Componant 1', fontsize=15)
    ax.set_ylabel('Componant 2', fontsize=15)
    ax.set_title('PCA en deux dimensions', fontsize=20)
    targets = donnees_B['DisplayName']

    for target in targets:
        indicesToKeep = finalDf['DisplayName'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'Componant 1']
                   , finalDf.loc[indicesToKeep, 'Componant 2']
                   , s=50)
        ax.annotate(target, (finalDf.loc[indicesToKeep, 'Componant 1'], finalDf.loc[indicesToKeep, 'Componant 2']))
    ax.grid()
    plt.show()


pca_en_2D(X)

def pca_en_5D(donnees):
    model = PCA(n_components=5)

    pca = model.fit_transform(donnees)


    principalDf = pd.DataFrame(data=pca
                               , columns=['Componant 1', 'Componant 2', 'Componant 3', 'Componant 4', 'Componant 5'])

    print(principalDf)


pca_en_5D(X)
