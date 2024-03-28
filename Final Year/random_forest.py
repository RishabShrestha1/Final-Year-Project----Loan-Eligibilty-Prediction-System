import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        unique_classes, counts = np.unique(y, return_counts=True)

        if len(unique_classes) == 1 or depth == self.max_depth:
            return {'value': unique_classes[0]}

        feature_index, threshold = self.get_best_split(X, y)
        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices

        left_subtree = self.fit(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self.fit(X[right_indices], y[right_indices], depth + 1)

        return {'feature_index': feature_index, 'threshold': threshold,
              'left': left_subtree, 'right': right_subtree}

    def get_best_split(self, X, y):
        best_info_gain = -1
        best_feature, best_threshold = None, None

        for feature_index in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_index])

            for threshold in unique_values:
                info_gain = self.calculate_information_gain(X, y, feature_index, threshold)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def calculate_information_gain(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices

        left_entropy = self.calculate_entropy(y[left_indices])
        right_entropy = self.calculate_entropy(y[right_indices])
        parent_entropy = self.calculate_entropy(y)

        weight_left = len(y[left_indices]) / len(y)
        weight_right = len(y[right_indices]) / len(y)

        information_gain = parent_entropy - (weight_left * left_entropy + weight_right * right_entropy)
        return information_gain

    def calculate_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def predict_single(self, node, sample):
        if 'value' in node:
            return node['value']

        if sample[node['feature_index']] <= node['threshold']:
            return self.predict_single(node['left'], sample)
        else:
            return self.predict_single(node['right'], sample)

    def predict(self, X):
        return [self.predict_single(self.tree, sample) for sample in X]

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None,random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state=random_state
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            tree.tree = tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.round(np.mean(predictions, axis=0))
    
    def predict_proba(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)
