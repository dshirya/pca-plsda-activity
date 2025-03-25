import numpy as np
import pandas as pd
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from evaluation_metrics import accuracy_score, f1_score
import plotly.graph_objects as go

# Import your data-loading function from your existing PLS‑DA module.
from pls_da_man import load_and_prepare_data

def _evaluate_subset(X, y, selected_features, n_components=2, scoring='accuracy'):
    """
    Helper: evaluates a candidate subset via 5‑fold CV.
    Scales data, fits a PLSRegression model on one‑hot encoded y, predicts, and computes the score.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx][selected_features]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx][selected_features]
        y_test = y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # One-hot encode training target for fitting
        y_train_dummies = pd.get_dummies(y_train)
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_train_scaled, y_train_dummies)
        
        y_pred_cont = pls.predict(X_test_scaled)
        predicted_indices = np.argmax(y_pred_cont, axis=1)
        y_test_dummies = pd.get_dummies(y_test)
        predicted_labels = [y_test_dummies.columns[i] for i in predicted_indices]
        
        if scoring == 'accuracy':
            score_val = accuracy_score(y_test.reset_index(drop=True), predicted_labels)
        else:
            score_val = f1_score(y_test.reset_index(drop=True), predicted_labels, average='macro')
        scores.append(score_val)
    return np.mean(scores)

def forward_selection_plsda(filepath, target_column='Class', max_features=None, n_components=2, 
                            scoring='accuracy', verbose=False, visualize=False):
    """
    Forward selection for PLS‑DA.
    
    Loads your CSV via load_and_prepare_data (so your DataFrame contains features and a "Class" column).
    If n_components > 1, it randomly initializes with n_components features.
    Then it iteratively adds features as long as the CV performance (accuracy or F1) does not drop.
    
    Parameters:
      filepath (str): Path to your CSV (e.g., "1929_Mendeleev_features_binary1.csv").
      target_column (str): Name of the target column (e.g., "Class").
      max_features (int or None): Maximum number of features to select.
      n_components (int): Number of PLS‑DA components.
      scoring (str): "accuracy" or "f1".
      verbose (bool): Print step‑by‑step details if True.
      visualize (bool): Show interactive performance plot if True.
      
    Returns:
      selected_features (list): List of selected feature names.
      performance_history (list): CV scores at each iteration.
    """
    # Load data (using your existing function)
    data_clean, numeric_data, target_data = load_and_prepare_data(filepath, target_column)
    if numeric_data is None:
         print("Error loading data.")
         return None, None

    features = list(numeric_data.columns)
    remaining = features.copy()
    selected = []
    performance_history = []
    iterations_info = []  # store (iteration, selected_features, score)
    iteration = 0

    if max_features is not None and max_features < n_components:
        raise ValueError("max_features must be at least as large as n_components.")

    if scoring not in ['accuracy', 'f1']:
        raise ValueError("Unsupported scoring metric. Use 'accuracy' or 'f1'.")

    # Randomly initialize with n_components features if applicable
    if n_components > 1:
        if len(remaining) < n_components:
            raise ValueError("Not enough features to initialize with n_components features.")
        initial_selected = random.sample(remaining, n_components)
        for feat in initial_selected:
            remaining.remove(feat)
        selected.extend(initial_selected)
        base_score = _evaluate_subset(numeric_data, target_data, selected, n_components=n_components, scoring=scoring)
        performance_history.append(base_score)
        iteration += 1
        iterations_info.append((iteration, selected.copy(), base_score))
        if verbose:
            print(f"Iteration {iteration}: Initial selected = {selected}, Score = {base_score:.4f}")
    else:
        base_score = -np.inf

    best_score = base_score

    # Iteratively add features
    while remaining and (max_features is None or len(selected) < max_features):
        best_candidate = None
        best_candidate_score = -np.inf
        for feat in remaining:
            trial_set = selected + [feat]
            if len(trial_set) < n_components:
                continue
            score_candidate = _evaluate_subset(numeric_data, target_data, trial_set, n_components=n_components, scoring=scoring)
            if score_candidate > best_candidate_score:
                best_candidate_score = score_candidate
                best_candidate = feat
        if best_candidate is not None and best_candidate_score >= best_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            best_score = best_candidate_score
            performance_history.append(best_score)
            iteration += 1
            iterations_info.append((iteration, selected.copy(), best_score))
            if verbose:
                print(f"Iteration {iteration}: Added '{best_candidate}', Selected = {selected}, Score = {best_score:.4f}")
        else:
            break

    if verbose:
        print("\nFinal selected features:", selected)
    if visualize:
        iterations = [info[0] for info in iterations_info]
        scores_plot = [info[2] for info in iterations_info]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iterations, y=scores_plot, mode='lines+markers', name='Performance'))
        fig.update_layout(title="Forward Selection Performance",
                          xaxis_title="Number of Features",
                          yaxis_title=f"CV {scoring.capitalize()} Score",
                          template="plotly_white")
        fig.show()
    
    return selected, performance_history

def backward_elimination_plsda(filepath, target_column='Class', min_features=1, n_components=2, 
                               scoring='accuracy', verbose=False, visualize=False):
    """
    Backward elimination for PLS‑DA.
    
    Loads your data via load_and_prepare_data and starts with all features.
    It then iteratively removes one feature at a time as long as the CV performance does not drop.
    
    Parameters:
      filepath (str): Path to your CSV file.
      target_column (str): Name of the target column.
      min_features (int): Minimum number of features to keep.
      n_components (int): Number of PLS‑DA components.
      scoring (str): "accuracy" or "f1".
      verbose (bool): Print step‑by‑step details if True.
      visualize (bool): Show interactive performance plot if True.
      
    Returns:
      remaining_features (list): List of features left after elimination.
      performance_history (list): CV scores at each elimination step.
    """
    data_clean, numeric_data, target_data = load_and_prepare_data(filepath, target_column)
    if numeric_data is None:
         print("Error loading data.")
         return None, None

    current_features = list(numeric_data.columns)
    performance_history = []
    iterations_info = []
    iteration = 0

    best_score = _evaluate_subset(numeric_data, target_data, current_features, n_components=n_components, scoring=scoring)
    performance_history.append(best_score)
    iterations_info.append((iteration, current_features.copy(), best_score))
    if verbose:
        print(f"Iteration {iteration}: All features, Score = {best_score:.4f}")
    
    while len(current_features) > min_features:
        best_score_after_removal = -np.inf
        feature_to_remove = None
        for feat in current_features:
            trial_set = [f for f in current_features if f != feat]
            score_trial = _evaluate_subset(numeric_data, target_data, trial_set, n_components=n_components, scoring=scoring)
            if score_trial > best_score_after_removal:
                best_score_after_removal = score_trial
                feature_to_remove = feat
        if feature_to_remove is not None and best_score_after_removal >= best_score:
            current_features.remove(feature_to_remove)
            best_score = best_score_after_removal
            performance_history.append(best_score)
            iteration += 1
            iterations_info.append((iteration, current_features.copy(), best_score))
            if verbose:
                print(f"Iteration {iteration}: Removed '{feature_to_remove}', Remaining = {current_features}, Score = {best_score:.4f}")
        else:
            break

    if verbose:
        print("\nFinal remaining features:", current_features)
    if visualize:
        # Instead of using iterations, compute the number of features remaining at each step.
        feature_counts = [len(info[1]) for info in iterations_info]
        scores_plot = [info[2] for info in iterations_info]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feature_counts, y=scores_plot, mode='lines+markers', name='Performance'))
        fig.update_layout(title="Backward Elimination Performance",
                        xaxis_title="Number of Features",
                        yaxis_title=f"CV {scoring.capitalize()} Score",
                        template="plotly_white")
        fig.show()
    
    return current_features, performance_history