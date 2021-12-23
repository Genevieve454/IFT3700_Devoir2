from pomegranate import BayesianNetwork, DiscreteDistribution, ConditionalProbabilityTable, Node
import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
import pygraphviz as pgv

donnees_C = pd.read_csv('../data/dataC.csv')
donnees_C = donnees_C.drop(columns=['DisplayName'])
colonnes = list(donnees_C)

liste_colonnes_utilises = []

# BayesianNetwork.from_samples()

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

        # don't process those who are now a children of another
        if _max_corr[columnNum] != -1 and not hasParentNotProcessed(columnNum, _max_corr):
            node = TreeNode(columnNum, [])
            addChildCorrelation(node, columnNum, _max_corr)
            _tree.append(node)

    return _tree


tree = buildCorrelationTree(ordre, max_corr)


# to test the tree with an output
def printTree(_tree, space=""):
    for _treeNode in _tree:
        print(space + str(_treeNode.columnNum))
        if len(_treeNode.children) > 0:
            for _node in _treeNode.children:
                printTree([_node], space + "  ")


printTree(tree)


def buildSubProbabilityTable(_node, _parentNode, _parentBayesianNode, _network, distribution):

    # We don't need to build a real probability table to show the diagram
    cp = ConditionalProbabilityTable(
        [[False, False, 0.95],
         [False, True, 0.05],

         [True, False, 0.1],
         [True, True, 0.9],
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


bayesnet = BayesianNetwork("Network")
bayesnet = buildProbabilityTable(tree, bayesnet)

bayesnet.bake()
plt.figure(figsize=(50,25))
bayesnet.plot()
plt.show()
