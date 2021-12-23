from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node
import pandas as pd
import json
from matplotlib import pyplot as plt

donnees_C = pd.read_csv('../data/dataC.csv')
donnees_C = donnees_C.drop(columns=['DisplayName'])
colonnes = list(donnees_C)

with open("../data/ordre.json") as fichier:
    ordre = json.load(fichier)

with open("../data/max_corr.json") as fichier:
    max_corr = json.load(fichier)


class TreeNode:
    def __init__(self, columnNum, children):
        self.columnNum = columnNum
        self.children = children


def addChildCorrelation(node, columnNum, _max_corr):
    for idx, val in enumerate(_max_corr):
        if columnNum == val:
            if not checkForSiblingCorrelation(columnNum, idx, _max_corr):
                _max_corr[idx] = -1  # removes the number from the list
                childNode = addChildCorrelation(TreeNode(idx, []), idx, _max_corr)
                node.children.append(childNode)

    return node


# Check if there's an infinite loop between two correlation
def checkForSiblingCorrelation(columnNum1, columnNum2, _max_corr):
    isSibling = False

    if _max_corr[columnNum2] == columnNum1 and _max_corr[columnNum1] == columnNum2:
        isSibling = True

    return isSibling


def hasParentNotProcessed(columnNum, _max_corr):
    hasParent = False

    potentialSibling = _max_corr[columnNum]
    isSibling = checkForSiblingCorrelation(columnNum, potentialSibling, _max_corr)
    hasParent = not isSibling

    return hasParent


def buildCorrelationTree(_ordre, _max_corr, _tree=[]):
    for columnNum in _ordre:

        if _max_corr[columnNum] != -1 and not hasParentNotProcessed(columnNum, _max_corr):
            node = TreeNode(columnNum, [])
            addChildCorrelation(node, columnNum, _max_corr)
            _tree.append(node)

    return _tree


tree = buildCorrelationTree(ordre, max_corr)


def buildSubProbabilityTable(_node, _parentNode, _parentBayesianNode, _network, distribution):

    nbRows = len(donnees_C)
    nbNonZeroA = donnees_C[colonnes[_node.columnNum]].astype(bool).sum(axis=0)
    nbZeroA = nbRows - nbNonZeroA

    nbNonZeroB = donnees_C[colonnes[_parentNode.columnNum]].astype(bool).sum(axis=0)
    nbZeroB = nbRows - nbNonZeroB

    cp = ConditionalProbabilityTable(
        [[False, False, (nbZeroA + nbZeroB) / (nbRows * 2)],
         [False, True, (nbZeroA + nbNonZeroB) / (nbRows * 2)],

         [True, False, (nbNonZeroA + nbZeroB) / (nbRows * 2)],
         [True, True, (nbNonZeroA + nbNonZeroB) / (nbRows * 2)],
         ],
        [distribution])

    bayesianNode = Node(cp, colonnes[_node.columnNum])
    _network.add_node(bayesianNode)
    _network.add_edge(_parentBayesianNode, bayesianNode)

    for child in _node.children:
        buildSubProbabilityTable(child, _node, bayesianNode, _network, cp)

    return _network


def buildProbabilityTable(_tree, _network):

    nbRows = len(donnees_C)
    for node in tree:
        nbNonZero = donnees_C[colonnes[node.columnNum]].astype(bool).sum(axis=0)
        nbZero = nbRows - nbNonZero
        dd = DiscreteDistribution({False: nbZero/nbRows, True: nbNonZero/nbRows})
        bayesianNode = Node(dd, colonnes[node.columnNum])
        _network.add_node(bayesianNode)

        for sub in node.children:
            _network = buildSubProbabilityTable(sub, node, bayesianNode, _network, dd)

    return _network


bayesnet = BayesianNetwork("Réseau")
bayesnet = buildProbabilityTable(tree, bayesnet)

bayesnet.bake()
plt.figure(figsize=(50, 25))
bayesnet.plot()
plt.show()

with open('../data/reseau.json', 'w') as f:
    f.write(bayesnet.to_json())
