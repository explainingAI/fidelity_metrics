""" Module for computing local explanations for a given image and model (Decision Tree or
Linear Regression).

Written by Miquel Miró Nicolau, (UIB) 2023.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Global scaler for normalizing explanation values
scaler = MinMaxScaler()

def dt_local_explanation(estim, image: np.array):
    """ Compute a local explanation for a given image using a Decision Tree model.

    The local explanation is derived based on the path from the root node to a leaf node in the
    decision tree. Each feature involved in the path contributes to the prediction, with its
    importance quantified by the reduction in impurity at each split.

    Args:
        estim: A Scikit-learn Decision Tree estimator.
        image (np.ndarray): A 1D Numpy array representing the image to be explained.

    Returns:
        np.ndarray: A 1D array representing the feature importance for the given image.
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

    # Normalize feature importance values and populate explanation vector
    total_importance = sum(importance.values())
    for feature_idx, value in importance.items():
        expl[feature_idx] = value / total_importance

    return expl

def lr_local_explanation(estim, image: np.array):
    """ Compute a local explanation for a given image using a Linear Regression model.

    The explanation is derived by scaling the feature contributions (weights multiplied
    by feature values) using Min-Max scaling.

    Args:
        estim: A Scikit-learn Linear Regression estimator.
        image (np.ndarray): A 1D Numpy array representing the image to be explained.

    Returns:
        np.ndarray: A scaled 1D array representing the feature contributions.
    """
    # Compute feature contributions as the product of weights and input features
    contributions = estim.coef_ * image

    # Normalize contributions to a range [0, 1]
    expl = scaler.fit_transform(contributions.reshape(-1, 1)).flatten()

    return expl