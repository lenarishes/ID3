import collections
import math
import Tree

def checkEqual(lst):
    return lst[1:] == lst[:-1]
def mostFreq(lst):
    c = collections.Counter(lst)
    return c.most_common(1)[0]
def empty(seq):
    try:
        return all(map(empty, seq))
    except TypeError:
        return False

def getVals(S):
    """
    Returns a list of class labels for training set
    S: list of tuples, features X examples = training set
    """
    try:
        return [attr[-1] for attr in S]
    except TypeError:
        return []

class DTBuilder:

    def __init__(self, features):
        """features: dict of values for each non-categorical attribute"""
        self.features = features

    def info(training_set):
        """
        The information needed to identify the class of an element of a set
        """
        S = getVals(training_set)
        subsets = [[x for x in vals if x == c] for c in set(S)] #set(S) = possible labels
        distr = [len(s)/len(S) for s in subsets] #prob distribution of classes
        return sum([-1*x*math.log(x) for x in distr])

    def split(self, f, S):
        """
        partition of set S on the basis of the values of a non-categorical attribute f into sets S1, S2, .., Sn
        S: example matrix
        """
        return [[x for x in S if x[f] == val] for val in self.features[f]]

    def gain(self, f, S):
        """
        Computes information gain on training set S due to attribute f
        """
        subsets = self.split(f, S)
        weighted_info = [len(s)*info(s) for s in subsets]
        average = sum(weighted_info)/len(S) # info needed to identify an element of S after the value of attribute f has been obtained
        return info(S) - average

                
def ID3(train_ex, attrs):
    """
    Builds a Decision Tree
    train_ex: list of lists/tuples? (matrix) examples/features
    attrs: list of features that haven't been used in the current path
    """
    vals = getVals(train_ex) #list of class labels for the training examples
    #If set is empty, return a single node with value Failure;
    if empty(train_ex):
        return Tree()
    #If set consists of records with the same categorical value, return a single node with that value;
    if checkEqual(vals):
    #reached purity
        return Tree(vals[0])
    #If no features left, return a single node with the most frequent categorical value in records of set
    if empty(attrs):
        return Tree(mostFreq(vals))
    
    #Let D be the attribute with largest Gain(D,S) among features left;
    D = max(attrs, key = lambda X: self.gain(x, train_ex)) #????? how to pass the training set as param???
    
    #Let {dj| j=1,2, .., m} be the values of attribute D;
    #Let {Sj| j=1,2, .., m} be the subsets of S consisting of records with value dj for attribute D;
    root = Tree(D)
    for subset in self.split(D, train_ex):
        root.children.append(ID3(subset, attrs.remove(D)))
        #Return a tree with root labeled D and arcs labeled 
        #   d1, d2, .., dm going respectively to the trees 
        #ID3(R-{D}, C, S1), ID3(R-{D}, C, S2), .., ID3(R-{D}, C, Sm);
    return root


