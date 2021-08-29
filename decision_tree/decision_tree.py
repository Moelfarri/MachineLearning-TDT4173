import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

#Sources of inspiration for my solution:
#https://www.kaggle.com/natevegh/decision-tree-from-scratch-theory-code-explained#Entropy---How-does-the-decision-tree-make-decisions?
#https://www.youtube.com/watch?v=jVh5NA9ERDA&t=4s&ab_channel=PythonEngineers
#https://www.youtube.com/watch?v=Bqi7EFFvNOg&ab_channel=PythonEngineer


class TreeNode():
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None): #asterix to force caller to explictily write value="something"
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value
        
    def isLeafNode(self):
        return self.value is not None
    
class DecisionTree():
    def __init__(self,  min_sample_split=2, max_tree_depth=np.inf):
        self.root             = None
        self.max_tree_depth   = max_tree_depth
        self.min_sample_split = min_sample_split
        
        self.feature_columns  = None
        self.rules           = []
        
    def fit(self, X, y):
        """
        Generates a decision tree for classification
        
        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        self.feature_columns = np.array(X.columns)
        X = np.array(X).copy()
        y = np.array(y).copy()
        self.root = self.growTree(X,y)
    
    def growTree(self, X, y, current_depth=0):
        samples, features = X.shape
        
        #Termination conditions
        if current_depth >= self.max_tree_depth or samples < self.min_sample_split or len(np.unique(y)) == 1:
            most_common = np.unique(y, return_counts=True)[0][0]
            leaf_value =  most_common
            return TreeNode(value=leaf_value)
        
        #initiate features at random - will give different decision tree splits because the algorithm is greedy during information gain stage
        feature_idxs = np.random.choice(features, features, replace=False)
        
        
        #Select Best split at each node based on the best information gain
        best_gain = -np.inf
        split_idx, split_threshold = None, None
        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                info_gain = self.infoGain(X_column, threshold, y)

                if info_gain > best_gain:
                    best_gain = info_gain
                    split_idx = feature_idx
                    split_threshold = threshold      
       
        best_feature, best_threshold = split_idx, split_threshold
        
        
        left_idxs = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
        right_idxs = np.argwhere(X[:, best_feature] > best_threshold).flatten()
        
        
        left = self.growTree(X[left_idxs, :], y[left_idxs], current_depth + 1)
        right = self.growTree(X[right_idxs, :], y[right_idxs], current_depth + 1)
        return TreeNode(best_feature, best_threshold, left, right)
        
        
    def infoGain(self, X_column, threshold, y):
        
        # Parent node Entropy
        _, y_counts = np.unique(y, return_counts=True)
        parent_entropy = entropy(y_counts)

        # Split based on one of the feature thresholds
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        

        # compute the weighted avg. of the entropy for the children
        weight_left, weight_right = len(left_idxs)/len(y),  len(right_idxs)/len(y)
        
        _, y_counts_left  = np.unique(y[left_idxs], return_counts=True) 
        _, y_counts_right = np.unique(y[right_idxs], return_counts=True) 
        
        #Calculate Entropy of children nodes
        E_l, E_r = entropy( y_counts_left), entropy(y_counts_right)
        child_entropy = weight_left * E_l +  weight_right * E_r

        # information gain is entropy before split minus entropy after split
        information_gain = parent_entropy - child_entropy
        return information_gain
    
       
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.
            
        Returns:
            A length m vector with predictions
        """
        
        #Traverse the tree recursively
        #At each node look at the best split feature of the test feature vector x
        #and go left or right depending on x[feature_idx] <= threshold
        
        #when we reach the leaf node we return the stored most common class label
        X = np.array(X).copy()
        return np.array([self.traverseTree(data, self.root) for data in X])
        
    
    
    def traverseTree(self, data, node):
        temp_rules = []
        while True:
            if data[node.feature] <= node.threshold:
                node = node.left
            else: 
                node = node.right
                
            #Collecting all the rules during traversing
            if not node.isLeafNode():
                temp_rules.append((self.feature_columns[node.feature], data[node.feature]))
                
            
            if node.isLeafNode():
                #Saving the rules and the predicted answer when leafNode is reached
                self.rules.append((temp_rules, node.value))
                temp_rules = []
                
                return node.value
            
    
    def get_rules(self):
        """
        Returns the decision tree as a list of rules
        
        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label
        
            attr1=val1 ^ attr2=val2 ^ ... => label
        
        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        #Naive way to remove duplicate rules, my traverse algorithm could need a patchup..
        res = []
        for i in self.rules:
            if i not in res and i != ([],"Yes"):
                res.append(i)
        return res

    
    
# --- Some utility functions 
    
def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy
    
    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning
    
    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0
            
    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.
    
    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))



