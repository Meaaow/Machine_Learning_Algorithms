import numpy as np
from typing import List
from hw1_knn import KNN
# from sklearn.metrics import accuracy_score

# #######################
# from data import data_processing
# Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    child_entropy_list = []
    total_sum = 0
    for tup in branches:
        sum_entropy = 0
        sum_num = sum(tup)
        total_sum += sum_num
        for i in tup:
            if i!= 0:
                sum_entropy +=  (float(i/sum_num) * np.log2(float(i/sum_num)))
            else:
                sum_entropy += 0
        print(' se',sum_entropy, "sum_num", sum_num)
        child_entropy_list.append(sum_entropy * sum_num)
        print('child_entropy_list',child_entropy_list)


    temp_res = sum(child_entropy_list)
    print('tempres', temp_res, "total_sum", total_sum)
    ans = S + float(temp_res / total_sum)
    print('ans',ans)
    return ans
#Information_Gain(0.97,[[2, 5], [10, 3]])


    # raise NotImplementedError


# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    node = decisionTree.root_node
    if node.splittable:
        prune_node(decisionTree, node, X_test, y_test)

def prune_node(decisionTree, node, X_test, y_test):
    if node.splittable:
        for child in node.children:
            prune_node(decisionTree, child, X_test, y_test)
        y_pred = decisionTree.predict(X_test)
        accuracy_without_pruning = calculate_accuracy(y_pred, y_test)
        node_children = node.children
        node.children = []
        node.splittable = False
        y_pred = decisionTree.predict(X_test)
        accuracy_with_pruning = calculate_accuracy(y_pred, y_test)
        if accuracy_without_pruning >= accuracy_with_pruning :
            node.children = node_children
            node.splittable = True

def calculate_accuracy(y_pred, y_test):
    match_count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            match_count += 1

    return float(match_count/len(y_test))



# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(0, len(real_labels)):
        if predicted_labels[i] == 1:
            if predicted_labels[i] == real_labels[i]:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if predicted_labels[i] == real_labels[i]:
                true_negatives += 1
            else:
                false_negatives += 1

    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives)

    if true_positives + false_negatives == 0:
        recall = 0
    else:
        recall = true_positives / (true_positives + false_negatives)

    if precision + recall != 0:
        f1_score = float(2 * precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score

# usage example:
# y_true = [1, 1, 0, 1, 1]
# y_pred = [0, 1, 0, 0, 1]
#
# f1_score = f1_score(y_true, y_pred)
# print("F1 score: ", f1_score)
#
#     raise NotImplementedError
# point1 = [1.0, 2.0, 3.0]
# point2 = [4,6,5]
#TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:

    a, b = np.array(point1), np.array(point2)
    x = np.linalg.norm(a-b)

    return float(x)
    #raise NotImplementedError

#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    a, b = np.array(point1), np.array(point2)
    x = np.inner(a,b)
    return float(x)
    #raise NotImplementedError

#TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    a, b = np.array(point1), np.array(point2)
    x = -1 * np.exp(-0.5*np.inner(a-b,a-b))
    return float(x)
    # raise NotImplementedError


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    a, b = np.array(point1), np.array(point2)
    num = np.dot(a,b)
    den = np.linalg.norm(a)* np.linalg.norm(b)
    x = 1 - float(num/den)
    return float(x)
    # raise NotImplementedError


# distance_funcs = {
#    'euclidean': euclidean_distance,
#    'gaussian': gaussian_kernel_distance,
#    'inner_prod': inner_product_distance,
#    'cosine_dist': cosine_sim_distance,
# }


# TODO: select an instance of KNN with the best f1 score on validation dataset

def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model

    distance_funcs_list=['euclidean','gaussian','inner_prod','cosine_dist']
    best_f1 = 0
    best_model = None
    best_k = -1
    best_func = "*"
    for k in range(1,30,2):
        if k < len(Xtrain):
            for func_string in distance_funcs_list:
                knn = KNN(k, distance_funcs[func_string])
                knn.train(Xtrain, ytrain)
                predicted_vals = knn.predict(Xval)
                curr_f1 = f1_score(yval, predicted_vals)
                if curr_f1 >  best_f1:
                    best_f1 = curr_f1
                    best_model = knn
                    best_k = k
                    best_func = func_string
    # print(best_model, best_k, best_func)
    return best_model, best_k, best_func


    # raise NotImplementedError

# model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval)


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    # raise NotImplementedError

    distance_funcs_list = ['euclidean','gaussian','inner_prod','cosine_dist']
    scalar_funcs_list = ['min_max_scale','normalize']
    best_f1 = 0
    best_model = None
    best_k = -1
    best_func = "*"
    best_scalar = "+"
    for k in range(1,30,2):
        for func_string in distance_funcs_list:
            for scaling_string in scalar_funcs_list:
                if k < len(Xtrain):
                    scalar_object = scaling_classes[scaling_string]()
                    scaled_Xtrain = scalar_object(Xtrain)
                    scaled_Xval = scalar_object(Xval)
                    knn = KNN(k, distance_funcs[func_string])
                    knn.train(scaled_Xtrain, ytrain)
                    predicted_vals = knn.predict(scaled_Xval)
                    curr_f1 = f1_score(yval, predicted_vals)
                    if curr_f1 >  best_f1:
                        best_f1 = curr_f1
                        best_model = knn
                        best_k = k
                        best_func = func_string
                        best_scalar = scaling_string
    print(best_model, best_k, best_func, best_scalar)
    return best_model, best_k, best_func, best_scalar

class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalised_Features = []
        for point in features:
            norm_point = []
            den = np.sqrt(inner_product_distance(point,point))
            if den != 0:
                for cood in point:
                    norm_point.append(float(cood/den))
            else:
                norm_point = [0]*len(point)
            normalised_Features.append(norm_point)
        return normalised_Features
        # raise NotImplementedError

# xx=NormalizationScaler()
# yy=xx.__call__([[3, 4], [1, -1], [0, 0]])
# print("y",yy)


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        self.max_val = None
        self.min_val = None
        # pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        features_np = np.array(features)

        if self.max_val is None or self.min_val is None:
            self.max_val = np.amax(features_np, axis = 0)
            self.min_val = np.amin(features_np, axis = 0)

        scaled_features = (features_np - self.min_val)/(self.max_val - self.min_val)

        return scaled_features.tolist()

        # raise NotImplementedError

# xx=MinMaxScaler()
# yy=xx.__call__([[2, -1], [-1, 5], [0, 0]])
# print(yy)
