""" Module for computing local explanation for a given image and a given Decision Tree.

Written by Miquel Miró Nicolau, UIB 2023
"""
import numpy as np


def local_explanation(estim, image: np.array):
    """Method for computing local explanation for a given image and a given Decision Tree.

    To obtain the local explanation, we start from the fact that the prediction of a decision tree
    is defined for the path from the root node to a leaf node, and that this path is selected
    analysing at each level a single feature, we proposed to set each of these features as important
    for the prediction. Finally, to quantify this importance, we toke into account the impurity
    criterion.

    Args:
        estim: Scikit-learn Decision Tree estimator.
        image: Numpy array representing the image to be explained.

    Returns:
        expl: Numpy array representing the local explanation for the given image and the given
    """
    children_left = estim.tree_.children_left
    children_right = estim.tree_.children_right
    feature = estim.tree_.feature
    threshold = estim.tree_.threshold
    impurity = estim.tree_.impurity

    importance = {}

    node_id = 0
    while children_left[node_id] != children_right[node_id]:

        if feature[node_id] not in importance:
            importance[feature[node_id]] = 0

        if image[feature[node_id]] <= threshold[node_id]:
            children_id = children_left[node_id]
        else:
            children_id = children_right[node_id]

        importance[feature[node_id]] += impurity[node_id] - impurity[children_id]
        node_id = children_id

    expl = np.zeros_like(image).astype(np.float64)

    adder = 0
    for feature, value in importance.items():
        adder += value
        expl[feature] = value

    expl /= adder

    return expl
