import numpy as np
from sklearn.metrics import accuracy_score as sk_accuracy_score, f1_score as sk_f1_score, confusion_matrix as sk_confusion_matrix, silhouette_score

def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
    Returns:
        accuracy (float): Proportion of correct predictions (0 to 1).
    """
    # Use sklearn's accuracy_score for consistency
    return sk_accuracy_score(y_true, y_pred)

def f1_score(y_true, y_pred, average='macro'):
    """
    Compute the F1-score for classification.
    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        average (str): Type of averaging for multi-class ('macro' by default).
    Returns:
        f1 (float): F1-score.
    """
    return sk_f1_score(y_true, y_pred, average=average)

def silhouette_score_latent(X_latent, y):
    """
    Compute the silhouette score for clustered data.
    Parameters:
        X_latent (array-like): Data points in latent space (e.g., PLS components).
        y (array-like): Cluster labels for each data point.
    Returns:
        silhouette (float): Silhouette score (ranges from -1 to 1).
    """
    # Silhouette score requires numeric labels, ensure y is integer encoded
    y = np.array(y)
    # If y is not numeric, convert to numeric labels
    if y.dtype == object or y.dtype.type is np.str_:
        unique_classes = np.unique(y)
        # map classes to integers
        class_map = {cls: idx for idx, cls in enumerate(unique_classes)}
        y = np.array([class_map[cls] for cls in y])
    return silhouette_score(X_latent, y)

def fisher_ratio(X_latent, y):
    """
    Calculate Fisher's Discriminant Ratio (FDR) for latent features.
    This ratio is (between-class variance) / (within-class variance) for the latent space.
    Parameters:
        X_latent (array-like): Data points in latent space (n_samples x n_components).
        y (array-like): Class labels for each data point.
    Returns:
        fdr (float): Fisher's discriminant ratio.
    """
    X_latent = np.array(X_latent)
    y = np.array(y)
    classes = np.unique(y)
    overall_mean = np.mean(X_latent, axis=0)
    between_class_variance = 0.0
    within_class_variance = 0.0
    for cls in classes:
        class_data = X_latent[y == cls]
        if class_data.size == 0:
            continue
        class_mean = np.mean(class_data, axis=0)
        # between-class variance contribution
        between_class_variance += class_data.shape[0] * np.sum((class_mean - overall_mean) ** 2)
        # within-class variance contribution
        within_class_variance += np.sum((class_data - class_mean) ** 2)
    # Avoid division by zero
    if within_class_variance == 0:
        return np.inf
    return between_class_variance / within_class_variance

def pairwise_class_distances(X_latent, y):
    """
    Compute pairwise Euclidean distances between class centroids in latent space.
    Parameters:
        X_latent (array-like): Data points in latent space (n_samples x n_components).
        y (array-like): Class labels for each data point.
    Returns:
        distances (dict): Dictionary with keys as tuple of class pair and values as distance.
    """
    X_latent = np.array(X_latent)
    y = np.array(y)
    classes = np.unique(y)
    centroids = {}
    for cls in classes:
        class_data = X_latent[y == cls]
        if class_data.size == 0:
            continue
        centroids[cls] = np.mean(class_data, axis=0)
    distances = {}
    class_list = list(centroids.keys())
    for i in range(len(class_list)):
        for j in range(i+1, len(class_list)):
            cls_i = class_list[i]
            cls_j = class_list[j]
            # Compute Euclidean distance
            dist = np.linalg.norm(centroids[cls_i] - centroids[cls_j])
            distances[(cls_i, cls_j)] = dist
    return distances

def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix.
    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
    Returns:
        cm (2D array): Confusion matrix counts with shape (n_classes, n_classes).
        classes: Array of class labels in order corresponding to matrix.
    """
    # Use sklearn's confusion_matrix to get counts
    cm = sk_confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate((np.array(y_true), np.array(y_pred))))
    return cm, classes