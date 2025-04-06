import numpy as np
from collections import Counter
class Node:
    def __init__(self,left = None,right = None, threshold = None,feature_index = None, value = None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature_index = feature_index

        self.value = value

    def is_leafNode(self):
        return self.value is not None

class DecisionTree_Classifier:
    def __init__(self,min_sample_split = 2, max_depth = 100,n_features = None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, Y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._build_tree(X, Y)

    def _build_tree(self, X, Y, depth=0):
        No_samples , No_features = X.shape
        n_labels = len(np.unique(Y))
        #check the stopping criteria
        if (depth>= self.max_depth or n_labels == 1 or No_samples<self.min_sample_split):
            leaf_value = self._most_common_label(Y)
            return Node(value = leaf_value)

        feature_index = np.random.choice(No_features, self.n_features, replace=False)

        #find the best split
        best_threshold, best_feature = self._best_split(X,Y,feature_index)

        #create child nodes and call func recursively 
        left_index, right_index = self._split(X[:,best_feature], best_threshold)
        left = self._build_tree(X[left_index, :], Y[left_index], depth+1)
        right = self._build_tree(X[right_index, :], Y[right_index], depth+1)

        return Node(left,right,best_threshold,best_feature)


    def _best_split(self, X, Y, feature_index):
        best_gain = -1
        split_index, split_threshold = None, None

        for feature in feature_index:
            X_column = X[:, feature]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                #calculate the information gain
                gain = self._information_gain(Y,X_column, thr)
                if gain>best_gain:
                    best_gain = gain
                    split_index = feature
                    split_threshold = thr
        return split_threshold, split_index

    def _information_gain(self,Y,X_column,thr):
        # for parent
        parent_entropy = self._entropy(Y)

        #childe entropy
        left_index, right_index = self._split(X_column, thr)

        if len(left_index) == 0 or len(right_index) == 0:
            return 0
        n = len(Y)
        n_l, n_r = len(left_index), len(right_index)
        e_l, e_r = self._entropy(Y[left_index]), self._entropy(Y[right_index])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        #calculate information gain
        information_gain = parent_entropy - child_entropy

        return information_gain

    def _split(self, X_column, split_threshold):
        left_index = np.argwhere(X_column<= split_threshold).flatten()
        right_index = np.argwhere(X_column> split_threshold).flatten()
        return left_index, right_index


    def _entropy(self, Y):
        hist = np.bincount(Y)
        ps = hist/len(Y)
        return -np.sum([p* np.log(p) for p in ps if p>0])  
        
    def _most_common_label(self, Y):
        counter = Counter(Y)
        value = counter.most_common(1)[0][0]
        return value


    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leafNode():
            return node.value
        
        if x[node.feature_index] <= node.threshold :
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)
        


class DecisionTree_Regressor:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(np.array(X), np.array(y))

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape

        if (depth >= self.max_depth or num_samples < self.min_samples_split 
            or len(np.unique(y)) == 1):
            leaf_value = np.mean(y)
            return Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            return Node(value=np.mean(y))

        # Create boolean masks for indexing
        left_index = X[:, best_feature] <= best_threshold
        right_index = X[:, best_feature] > best_threshold

        # Use boolean masks for indexing
        left = self._build_tree(X[left_index], y[left_index], depth + 1)
        right = self._build_tree(X[right_index], y[right_index], depth + 1)

        return Node(left, right, best_threshold, best_feature)

    def _mse(self, y):
        #Calculate mean squared error (variance) for regression
        if len(y) <= 1:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _variance_reduction(self, y, left_y, right_y):
        #Calculate variance reduction for split evaluation
        parent_var = self._mse(y)
        n = len(y)
        n_l, n_r = len(left_y), len(right_y)
        
        if n_l == 0 or n_r == 0:
            return 0
            
        # Calculate weighted average of children variance
        child_var = (n_l / n) * self._mse(left_y) + (n_r / n) * self._mse(right_y)
        
        # Variance reduction
        return parent_var - child_var

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                # Create boolean masks for the split
                left_index = X_column <= threshold
                right_index = X_column > threshold
                
                # Skip if split doesn't create meaningful partitions
                if np.sum(left_index) < 1 or np.sum(right_index) < 1:
                    continue
                
                # Calculate variance reduction
                gain = self._variance_reduction(y, y[left_index], y[right_index])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leafNode():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)