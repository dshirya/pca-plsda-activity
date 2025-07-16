import random
import warnings
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .helpers import clean_array


def evaluate_subset(
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: list[str],
    n_components: int = 2,
    scoring: str = "accuracy"
) -> float:
    """
    Evaluate a feature subset via PLS-DA.
    
    Args:
        X: Feature matrix DataFrame
        y: Target Series
        selected_features: List of feature names to use
        n_components: Number of PLS components
        scoring: Scoring metric ('accuracy' or 'f1')
        
    Returns:
        Score for the feature subset
    """
    # 1) Slice to selected features
    X_sub = X[selected_features].copy()

    # 2) Drop zero-variance columns
    zero_var = X_sub.std(axis=0) == 0
    if zero_var.any():
        X_sub = X_sub.loc[:, ~zero_var]

    # 3) Check if enough features for PLS
    if X_sub.shape[1] < n_components:
        return np.nan

    # 4) Scale and clean any NaN/inf
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    X_scaled = clean_array(X_scaled)

    # 5) One-hot encode target
    Y_dummy = pd.get_dummies(y)

    # 6) Fit PLS model
    pls = PLSRegression(n_components=n_components, scale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            pls.fit(X_scaled, Y_dummy)
        except Exception:
            return np.nan

    # 7) Predict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            y_pred_cont = pls.predict(X_scaled)
        except Exception:
            return np.nan

    # 8) Get class predictions via argmax
    idxs = np.argmax(y_pred_cont, axis=1)
    preds = [Y_dummy.columns[i] for i in idxs]

    # 9) Calculate score
    if scoring == "accuracy":
        return accuracy_score(y, preds)
    else:
        return f1_score(y, preds, average="macro")


def forward_selection_plsda_df(
    numeric_data: pd.DataFrame,
    target_data: pd.Series,
    n_components: int = 2,
    plateau_steps: int = 10,
    init_features: list[str] | None = None,
    scoring: str = "accuracy"
) -> pd.DataFrame:
    """
    Greedy forward PLS-DA feature selection with plateau stopping.
    
    Args:
        numeric_data: DataFrame of features
        target_data: Series of target values
        n_components: Number of PLS components
        plateau_steps: Stop after this many additions without accuracy gain
        init_features: Optional list of features to start with
        scoring: Scoring metric to use
        
    Returns:
        DataFrame with selection history
    """
    all_feats = list(numeric_data.columns)
    remaining = all_feats.copy()
    selected: list[str] = []
    records: list[dict] = []

    iteration = 0
    best_score = -np.inf
    plateau_count = 0

    # Initial seeding
    if init_features:
        for f in init_features:
            if f not in remaining:
                raise ValueError(f"init_features contains '{f}', which is not in numeric_data.")
            remaining.remove(f)
        selected = init_features.copy()
        best_score = evaluate_subset(numeric_data, target_data, selected,
                                   n_components=n_components, scoring=scoring)
        iteration += 1
        records.append({
            'step': iteration,
            'accuracy': best_score,
            'feature_added': ','.join(init_features),
            'total_features': len(selected)
        })
    elif n_components > 1:
        # Random seed of size n_components
        seed = random.sample(remaining, n_components)
        for f in seed:
            remaining.remove(f)
        selected = seed.copy()
        best_score = evaluate_subset(numeric_data, target_data, selected,
                                   n_components=n_components, scoring=scoring)
        iteration += 1
        records.append({
            'step': iteration,
            'accuracy': best_score,
            'feature_added': ','.join(seed),
            'total_features': len(selected)
        })

    # Greedy forward selection with plateau stopping
    while remaining:
        # Find best single feature to add
        best_cand, best_cand_score = None, -np.inf
        for feat in random.sample(remaining, len(remaining)):
            trial = selected + [feat]
            if len(trial) < n_components:
                continue
            s = evaluate_subset(numeric_data, target_data, trial,
                              n_components=n_components, scoring=scoring)
            if s > best_cand_score:
                best_cand_score, best_cand = s, feat

        # Only add if it doesn't drop below current best
        if best_cand is None or best_cand_score < best_score:
            break

        # Plateau logic
        if best_cand_score == best_score:
            plateau_count += 1
        else:
            plateau_count = 0

        remaining.remove(best_cand)
        selected.append(best_cand)
        iteration += 1
        records.append({
            'step': iteration,
            'accuracy': best_cand_score,
            'feature_added': best_cand,
            'total_features': len(selected)
        })

        # Update best_score if improved
        if best_cand_score > best_score:
            best_score = best_cand_score

        # Stop if plateau too long
        if plateau_count >= plateau_steps:
            break

    return pd.DataFrame.from_records(records)


def backward_elimination_plsda_df(
    numeric_data: pd.DataFrame,
    target_data: pd.Series,
    min_features: int = 1,
    n_components: int = 2,
    scoring: str = "accuracy"
) -> pd.DataFrame:
    """
    Greedy backward elimination PLS-DA down to min_features.
    
    Args:
        numeric_data: DataFrame of features
        target_data: Series of target values
        min_features: Minimum number of features to keep
        n_components: Number of PLS components
        scoring: Scoring metric to use
        
    Returns:
        DataFrame with elimination history
    """
    # Start with all features
    current = list(numeric_data.columns)
    records: list[dict] = []
    iteration = 0

    # Evaluate the full set
    best_score = evaluate_subset(
        numeric_data, target_data, current,
        n_components=n_components, scoring=scoring
    )
    records.append({
        "step": iteration,
        "accuracy": best_score,
        "feature_removed": "",
        "total_features": len(current)
    })

    # Greedy elimination loop
    while len(current) > min_features:
        best_after = -np.inf
        feat_to_remove = None

        # Try removing each feature
        for feat in random.sample(current, len(current)):
            trial = [f for f in current if f != feat]
            if len(trial) < n_components:
                continue
            s = evaluate_subset(
                numeric_data, target_data, trial,
                n_components=n_components, scoring=scoring
            )
            if s > best_after:
                best_after, feat_to_remove = s, feat

        # If no removal can match or exceed current best, stop
        if feat_to_remove is None or best_after < best_score:
            break

        # Commit the removal
        current.remove(feat_to_remove)
        iteration += 1
        best_score = best_after
        records.append({
            "step": iteration,
            "accuracy": best_score,
            "feature_removed": feat_to_remove,
            "total_features": len(current)
        })

    return pd.DataFrame.from_records(records) 