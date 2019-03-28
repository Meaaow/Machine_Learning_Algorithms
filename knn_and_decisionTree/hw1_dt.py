import numpy as np
import utils as Util
# from sklearn.metrics import accuracy_score

#
# import data
# import hw1_dt as decision_tree
# X_test, y_test = data.sample_decision_tree_test()
#
# print("***********************************hw1_dt***************************************")

class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            print ("feature: ", feature)
            print ("pred: ", pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        unique_classes = sorted(np.unique(self.labels))
        # print("selflabels",self.labels)

        max_ig = 0
        def calculate_entropy():
            labels_and_count = sorted([(x, (self.labels).count(x)) for x in set((self.labels))], key=lambda y: y[1])
            total_sum = 0
            den = sum(x[1] for x in labels_and_count)
            for tup in labels_and_count:
                if tup[1] == 0:
                    total_sum += 0
                else:
                    total_sum += (-1*(tup[1]/den) * np.log2(float(tup[1]/den)))
            print('S .. entropy', total_sum)
            return total_sum

        S = calculate_entropy()

        for feat_no in range(len(self.features[0])):
            feature_dict = {}

            feature_column = [row[feat_no] for row in self.features]
            print("feature_column", feature_column)

            if None in feature_column:
                continue

            unique_vals = sorted(np.unique(feature_column))

            for val in unique_vals:
                for clas in unique_classes:
                    feature_dict[val] = feature_dict.get(val, {})
                    feature_dict[val][clas] = feature_dict[val].get(clas, 0)

            for val, clas in zip(feature_column, self.labels):
                feature_dict[val][clas] += 1
                # print(feature_dict)

            list_for_ig = []
            for key, val in feature_dict.items():
                list_branch = []
                for key2, val2 in val.items():
                    list_branch.append(val2)
                list_for_ig.append(list_branch)

            print(list_for_ig)

            ig = Util.Information_Gain(S, list_for_ig)
            print("decision tree --- ig ------>>> ", ig)

            if self.feature_uniq_split:
                unique_attributes = len(self.feature_uniq_split)
            else:
                unique_attributes = 0

            if ig > max_ig:
                max_ig = ig
                self.dim_split = feat_no
                self.feature_uniq_split = unique_vals
            elif ig == max_ig and len(unique_vals) > unique_attributes:
                max_ig = ig
                self.dim_split = feat_no
                self.feature_uniq_split = unique_vals

        XX = np.array(self.features)[:, self.dim_split]
        dtypeX = np.array(self.features, dtype=object)

        dtypeX[:, self.dim_split] = None

        for unique_value in self.feature_uniq_split:
            indexes = np.where(XX == unique_value)
            x_new = dtypeX[indexes].tolist()
            y_new = np.array(self.labels)[indexes].tolist()
            child = TreeNode(x_new, y_new, self.num_cls)
            if np.array(x_new).size == 0 or all(v is None for v in x_new[0]):
                child.splittable = False
            self.children.append(child)
        for child in self.children:
            if child.splittable:
                child.split()

        return

        # raise NotImplementedError



    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable:
            unique_value_list = np.array(self.feature_uniq_split).tolist()
            indexes = unique_value_list.index(feature[self.dim_split])
            return self.children[indexes].predict(feature)
        else:
            return self.cls_max
        # raise NotImplementedError





# # build the tree
# dTree = decision_tree.DecisionTree()
# dTree.train(features, labels)
#
# # print
# Util.print_tree(dTree)
#
# y_est_test = dTree.predict(X_test)
# test_accu = accuracy_score(y_est_test, y_test)
# print('test_accu', test_accu)
#
# print("done")
