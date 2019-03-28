from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        self.features = features
        self.labels = labels
        # raise NotImplementedError


    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        predicted_labels = []
        for point in features:
            maxcnt, itm = 0, 0
            neighbor_dict = {}
            point_kneighbors = self.get_k_neighbors(point)
            for neighbor in point_kneighbors:
                neighbor_dict[neighbor] = neighbor_dict.get(neighbor,0) + 1
                if neighbor_dict[neighbor] >= maxcnt:
                    maxcnt = neighbor_dict[neighbor]
                    itm = neighbor
            predicted_labels.append(itm)

        return predicted_labels

    #TODO: find KNN of one point

    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor

        k_labels, indexes = [99999] * self.k, [-1] * self.k

        for i, feature in enumerate(self.features):
            dis = self.distance_function(point, feature)

            if dis <= max(k_labels):
                max_index = np.argmax(k_labels)
                k_labels[max_index] = dis
                indexes[max_index] = i

        return np.take(self.labels, indexes)


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
