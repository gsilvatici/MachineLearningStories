from graphviz import Digraph
from .gain_functions import shannon, gini
import uuid


class DecisionTree:
    def __init__(self, training_set, config, simplify=False):
        GAIN_FUNCTION = {"shannon": shannon, "gini": gini}
        self.gain_function = GAIN_FUNCTION[config["gain_function"].lower()]
        self.max_depth = config.get('max_depth', float('inf'))

        if not self.gain_function:
            raise Exception("Invalid gain function")
        self.training_set = training_set
        self.objective = config["objective"]
        self.most_likely = int(training_set[self.objective].mode())
        self._tree = self.__generate_subtree(training_set, parent=None)

        if simplify:
            print(
                "WARNING: simplified option is still pretty much untested... Might produce bad trees"
            )
            self.__simplify_rec(self.tree)

    @property
    def tree(self):
        return self._tree

    def predict(self, input_set, debug=False, no_info_value=None):
        results = []
        if no_info_value is None:
            no_info_value = self.most_likely
        for item in input_set:
            results.append(self.tree.predict(item, debug, no_info_value=no_info_value))
        return results

    def digraph(self):
        dot = Digraph()
        self.__add_node_rec(dot, self.tree)
        return dot

    # For graphing only :)
    def __add_node_rec(self, dot, node, parent=None):
        name = node.value
        dot.node(node.id, str(name))
        if parent is not None:
            dot.edge(parent.id, node.id)

        for child in node.children:
            self.__add_node_rec(dot, child, node)

    def __simplify_rec(self, node):
        if node.is_leaf:
            return

        new_children = []
        new_children_set = set()
        for child in node.children:
            branches = self.__branches(child)
            if branches not in new_children_set:
                new_children_set.add(branches)
                new_children.append(child)

        # can't simplify, go deeper
        if set(node.children) == set(new_children) or len(set(new_children)) > 1:
            for child in node.children:
                self.__simplify_rec(child)

        # this means all children have the same subtree as their children
        else:
            parent = node.parent
            # remove current node
            parent.children.remove(node)
            # replace current node with children of children
            parent.children.extend(new_children[0].children)

    def __branches(self, node):
        branches_list = []
        self.__branches_rec(branches_list, f"{self.tree.value}", node)
        return tuple(branches_list)

    def __branches_rec(self, branches_list, branch, node):
        if node.is_leaf:
            branches_list.append(branch)
        else:
            for child in node.children:
                branch = f"{branch};{child.value}"
                self.__branches_rec(branches_list, branch, child)

    def __generate_subtree(self, data, parent, depth=0):
        classes = list(data.keys())
        objective = self.objective

        # Case (1 and 2) dataset has all values the same
        if len(data[objective].unique()) == 1:
            return Node(data[objective].unique()[0], parent)

        # Case (3) attributes are empty
        if len(classes) == 1 or depth == self.max_depth:
            return Node(str(data[objective].mode()[0]), parent)

        # remove objective class from classes so that only attributes remain
        classes.remove(objective)

        # Case (4) obtain root node
        # First calulate all possible gains
        gains_tuple = [(self.__gain(attr), attr) for attr in classes]
        # Choose attribute that maximizes Gain
        _, max_attr = max(gains_tuple)
        node = Node(max_attr, parent)
        depth += 1
        children = []

        # for each posible value in the winner attribute
        for value in data[max_attr].unique():
            # Generate subset where max_attr is value and drop that column
            edge = Node(str(value), node)
            children.append(edge)
            new_data = data[data[max_attr] == value].drop(max_attr, axis=1)
            edge.children.append(self.__generate_subtree(new_data, parent=edge, depth=depth))

        node.children = children
        return node

    def __gain(self, attribute):
        """
        Gain(S, A) = H(S) - sum( (|S_v|/|S|) * H(S_v) )
        where S_v = subset of S for each value of attribute A

        Returns: float
        """
        data = self.training_set

        def sv(df, attr, val):
            # Subset of df where column attr has values val
            return df[df[attr] == val]

        # Array of subsets for each possible value of column attribute
        subsets = [sv(data, attribute, v) for v in data[attribute].unique()]

        # H(S)
        general_gain = self.gain_function(data[self.objective])

        sum_gains = 0
        for s_v in subsets:
            # frequency of value v for column attribute
            frequency = len(s_v) / len(data)
            # H(S_v)
            gain_v = self.gain_function(s_v[self.objective])
            sum_gains += frequency * gain_v

        return general_gain - sum_gains


class Node:
    def __init__(self, value, parent=None):
        self.value = value
        self.parent = parent
        self.children = []
        self.id = str(uuid.uuid4())

    def __eq__(self, other):
        if type(self) == type(other):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    def predict(self, item, debug, no_info_value):
        if self.is_leaf:
            return self.value

        attributes = item.keys()
        # print(f"current node: {self.value}")
        # print(f"attributes: {attributes}")

        for attribute in attributes:
            if self.value == attribute:
                for child in self.children:
                    # print(f"childval: {child.value}; itemval: {item[attribute]}")
                    # print(f"childtype: {type(child.value)}; itemtype: {type(item[attribute])}")
                    if child.value == str(item[attribute]):
                        return child.children[0].predict(item, debug, no_info_value)
        if debug:
            print(f"Couldn't find attribute for item:\n {item}")
        return no_info_value

    @property
    def is_leaf(self):
        return len(self.children) == 0
