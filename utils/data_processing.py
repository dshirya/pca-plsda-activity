import pandas as pd
from config.settings import DATA_PATHS, LABEL_COLUMN


def load_features_data():
    """
    Load the features dataset.
    
    Returns:
        DataFrame with features data
    """
    return pd.read_excel(DATA_PATHS["features_binary"])


def load_pca_data():
    """
    Load the PCA dataset.
    
    Returns:
        DataFrame with elemental properties data
    """
    return pd.read_excel(DATA_PATHS["elemental_properties"])


def load_cluster_data():
    """
    Load the clustering dataset.
    
    Returns:
        DataFrame with Pauling data
    """
    return pd.read_excel(DATA_PATHS["pauling_data"])


def load_structure_data():
    """
    Load the structure type data for clustering visualization.
    
    Returns:
        DataFrame with Formula and Structure type columns
    """
    return pd.read_excel(DATA_PATHS["pauling_data"], usecols=["Formula", "Structure type"])


def prepare_plsda_data(df):
    """
    Prepare data for PLS-DA analysis.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN]
    return X, y


def filter_valid_features(df, selected_features):
    """
    Filter to only valid features that exist in the dataframe.
    
    Args:
        df: Input DataFrame
        selected_features: List of feature names
        
    Returns:
        List of valid feature names
    """
    return [f for f in selected_features if f in df.columns]


def remove_zero_variance_features(df):
    """
    Remove features with zero variance.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with zero-variance features removed
    """
    zero_var = df.std(axis=0) == 0
    if zero_var.any():
        return df.loc[:, ~zero_var]
    return df


def remove_nan_columns(df):
    """
    Remove columns that contain any NaN values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with NaN columns removed
    """
    return df.dropna(axis=1, how="any") 