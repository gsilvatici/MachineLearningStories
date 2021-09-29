import numpy as np

from .decision_tree import DecisionTree


class RandomForest:
    def __init__(self, df, cfg, sample_proportion=1, tree_count=1, replace=True):
        self.forest = []
        for _ in range(tree_count):
            subset = df.sample(frac=sample_proportion, replace=replace)
            # print(f"subset len: {len(subset)}")
            tree = DecisionTree(subset, cfg)
            self.forest.append(tree)

    def predict(self, input_set, debug=False):
        results = []
        for tree in self.forest:
            results.append(tree.predict(input_set))

        results = np.asarray(results)
        if debug:
            print(results)
        axis = 0
        u, indices = np.unique(results, return_inverse=True)
        return list(u[
            np.argmax(
                np.apply_along_axis(
                    np.bincount,
                    axis,
                    indices.reshape(results.shape),
                    None,
                    np.max(indices) + 1,
                ),
                axis=axis,
            )
        ])
