import os
import numpy as np
import pandas as pd
import plotly.express as px
from IPython.display import display, clear_output
import ipywidgets as widgets
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
# Import evaluation metrics from your external file
from pls_da.evaluation_metrics import accuracy_score, f1_score, silhouette_score_latent, fisher_ratio, pairwise_class_distances

# ====================================================
# Data Loading and Preparation
# ====================================================
def load_and_prepare_data(filepath, target_column='Class'):
    """
    Loads an Excel file, drops columns with missing values,
    extracts numeric features (excluding the target column),
    and returns the clean data, numeric data, and target (class) data.
    """
    try:
        data = pd.read_csv(filepath)
        data_clean = data.dropna(axis=1)
        if target_column not in data_clean.columns:
            print(f"Target column '{target_column}' not found in data.")
            return None, None, None

        target_data = data_clean[target_column]
        numeric_data = data_clean.select_dtypes(include=['float64', 'int64']).copy()
        # Exclude the target column if it appears as numeric
        if target_column in numeric_data.columns:
            numeric_data = numeric_data.drop(columns=[target_column])
        numeric_data.dropna(inplace=True)
        target_data = target_data.loc[numeric_data.index]
        return data_clean, numeric_data, target_data
    except Exception as e:
        print("Error loading file:", e)
        return None, None, None

# ====================================================
# PLS-DA Analysis
# ====================================================
def perform_plsda(numeric_data, selected_features, target_data):
    """
    Standardizes the selected numeric data, one-hot encodes the target,
    performs PLS-DA (PLSRegression with 2 components), and computes
    the explained variance percentages for each component.
    
    Returns:
      - pls: the fitted PLSRegression model
      - scores: the two-component PLS scores
      - pls1_percent: percentage variance explained by component 1
      - pls2_percent: percentage variance explained by component 2
    """
    X = numeric_data[selected_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # One-hot encode the target classes
    y = pd.get_dummies(target_data)
    pls = PLSRegression(n_components=2, scale=False)
    pls.fit(X_scaled, y)
    scores = pls.x_scores_
    
    # Compute explained variance ratios (approximate)
    total_variance = np.var(X_scaled, axis=0).sum()
    var_comp1 = np.var(scores[:, 0])
    var_comp2 = np.var(scores[:, 1])
    pls1_percent = var_comp1 / total_variance * 100
    pls2_percent = var_comp2 / total_variance * 100
    
    return pls, scores, pls1_percent, pls2_percent

# ====================================================
# Visualization
# ====================================================
def create_scatter_plot(pls_df, pls1_percent, pls2_percent):
    """
    Creates a Plotly scatter plot (using ggplot2 styling) of the PLS-DA results.
    The axis labels include the explained variance percentages.
    Colors are assigned from a custom list.
    """
    # Custom colors provided
    colors = [
        "#c3121e",  # Sangre
        "#0348a1",  # Neptune
        "#ffb01c",  # Pumpkin
        "#027608",  # Clover
        "#1dace6",  # Cerulean
        "#9c5300",  # Cocoa
        "#9966cc",  # Amethyst
        "#ff4500",  # Orange Red
    ]
    # Map each unique class to a color (using the first few colors)
    unique_classes = sorted(pls_df['Class'].unique())
    color_map = {cl: colors[i % len(colors)] for i, cl in enumerate(unique_classes)}
    
    fig = px.scatter(
        pls_df,
        x='Component1',
        y='Component2',
        color='Class',
        labels={
            "Component1": f"PLS Component 1 ({pls1_percent:.1f}%)",
            "Component2": f"PLS Component 2 ({pls2_percent:.1f}%)"
        },
        template="ggplot2",
        color_discrete_map=color_map
    )
    fig.update_traces(marker=dict(size=20, opacity=0.8))
    fig.update_layout(width=900, height=800, showlegend=True, font=dict(size=18))
    return fig

# ====================================================
# Build UI for Feature Selection with Logical Groups
# ====================================================
def build_plsda_ui(all_features, update_plsda_func, plsda_df_container):
    """
    Builds and returns a complete UI layout for controlling the PLS-DA analysis.
    Features are separated into logical groups with a group-specific toggle button.
    
    Parameters:
      - all_features: list of numeric feature names from the data.
      - update_plsda_func: callback function to update the PLS-DA analysis.
      - plsda_df_container: dict container to store latest results.
      
    Returns:
      An ipywidgets.VBox layout containing all UI elements.
    """
    # Define logical groups of features (only include features present in all_features)
    groups = {
        "Index Features": [
            "index_A", "index_B", "normalized_index_A", "normalized_index_B",
            "largest_index", "smallest_index", "avg_index"
        ],
        "Atomic Weight": [
            "atomic_weight_weighted_A+B", "atomic_weight_A/B", "atomic_weight_A-B"
        ],
        "Period & Group": [
            "period_A", "period_B", "group_A", "group_B", "group_A-B"
        ],
        "Mendeleev Numbers": [
            "Mendeleev_number_A", "Mendeleev_number_B", "Mendeleev_number_A-B"
        ],
        "Valence Electrons": [
            "valencee_total_A", "valencee_total_B", "valencee_total_A-B",
            "valencee_total_A+B", "valencee_total_weighted_A+B", "valencee_total_weighted_norm_A+B"
        ],
        "Unpaired Electrons": [
            "unpaired_electrons_A", "unpaired_electrons_B", "unpaired_electrons_A-B",
            "unpaired_electrons_A+B", "unpaired_electrons_weighted_A+B", "unpaired_electrons_weighted_norm_A+B"
        ],
        "Gilman": [
            "Gilman_A", "Gilman_B", "Gilman_A-B", "Gilman_A+B",
            "Gilman_weighted_A+B", "Gilman_weighted_norm_A+B"
        ],
        "Effective Nuclear Charge": [
            "Z_eff_A", "Z_eff_B", "Z_eff_A-B", "Z_eff_A/B",
            "Z_eff_max", "Z_eff_min", "Z_eff_avg", "Z_eff_weighted_norm_A+B"
        ],
        "Ionization Energy": [
            "ionization_energy_A", "ionization_energy_B", "ionization_energy_A-B", "ionization_energy_A/B",
            "ionization_energy_max", "ionization_energy_min", "ionization_energy_avg", "ionization_energy_weighted_norm_A+B"
        ],
        "Coordination & Ratio": [
            "coordination_number_A", "coordination_number_B", "coordination_number_A-B",
            "ratio_closest_A", "ratio_closest_B", "ratio_closest_max", "ratio_closest_min", "ratio_closest_avg"
        ],
        "Polyhedron Distortion": [
            "polyhedron_distortion_A", "polyhedron_distortion_B",
            "polyhedron_distortion_max", "polyhedron_distortion_min", "polyhedron_distortion_avg"
        ],
        "CIF Radius": [
            "CIF_radius_A", "CIF_radius_B", "CIF_radius_A/B", "CIF_radius_A-B",
            "CIF_radius_avg", "CIF_radius_weighted_norm_A+B"
        ],
        "Pauling Radius (CN12)": [
            "Pauling_radius_CN12_A", "Pauling_radius_CN12_B", "Pauling_radius_CN12_A/B",
            "Pauling_radius_CN12_A-B", "Pauling_radius_CN12_avg", "Pauling_radius_CN12_weighted_norm_A+B"
        ],
        "Pauling Electronegativity": [
            "Pauling_EN_A", "Pauling_EN_B", "Pauling_EN_A-B", "Pauling_EN_A/B",
            "Pauling_EN_max", "Pauling_EN_min", "Pauling_EN_avg", "Pauling_EN_weighted_norm_A+B"
        ],
        "Martynov Batsanov EN": [
            "Martynov_Batsanov_EN_A", "Martynov_Batsanov_EN_B", "Martynov_Batsanov_EN_A-B",
            "Martynov_Batsanov_EN_A/B", "Martynov_Batsanov_EN_max", "Martynov_Batsanov_EN_min",
            "Martynov_Batsanov_EN_avg", "Martynov_Batsanov_EN_weighted_norm_A+B"
        ],
        "Melting Point (K)": [
            "melting_point_K_A", "melting_point_K_B", "melting_point_K_A-B", "melting_point_K_A/B",
            "melting_point_K_max", "melting_point_K_min", "melting_point_K_avg", "melting_point_K_weighted_norm_A+B"
        ],
        "Density": [
            "density_A", "density_B", "density_A-B", "density_A/B",
            "density_max", "density_min", "density_avg", "density_weighted_norm_A+B"
        ],
        "Specific Heat": [
            "specific_heat_A", "specific_heat_B", "specific_heat_A-B", "specific_heat_A/B",
            "specific_heat_max", "specific_heat_min", "specific_heat_avg", "specific_heat_weighted_norm_A+B"
        ],
        "Cohesive Energy": [
            "cohesive_energy_A", "cohesive_energy_B", "cohesive_energy_A-B", "cohesive_energy_A/B",
            "cohesive_energy_max", "cohesive_energy_min", "cohesive_energy_avg", "cohesive_energy_weighted_norm_A+B"
        ],
        "Bulk Modulus": [
            "bulk_modulus_A", "bulk_modulus_B", "bulk_modulus_A-B", "bulk_modulus_A/B",
            "bulk_modulus_max", "bulk_modulus_min", "bulk_modulus_avg", "bulk_modulus_weighted_norm_A+B"
        ]
    }
    
    group_widgets = []
    all_feature_buttons = []  # To hold all toggle buttons globally
    
    # Build group widgets only for those features that are present in the data
    for group_name, feature_list in groups.items():
        # Filter features that exist in the dataset
        features_in_group = [feat for feat in feature_list if feat in all_features]
        if not features_in_group:
            continue
        
        # Create toggle buttons for each feature in this group
        buttons = [
            widgets.ToggleButton(
                value=True,
                description=feat,
                tooltip=f"Toggle feature: {feat}",
                layout=widgets.Layout(width='150px', height='30px')
            )
            for feat in features_in_group
        ]
        # Add these buttons to the global list
        all_feature_buttons.extend(buttons)
        
        # Create a "Toggle All" button for the group
        toggle_all_button = widgets.Button(
            description="Toggle",
            button_style='primary',
            layout=widgets.Layout(width='60px', height='25px', margin='5px')
        )
        def on_toggle_all_clicked(b, these_buttons=buttons):
            # If all are on, turn them off; otherwise, turn them all on.
            if all(btn.value for btn in these_buttons):
                for btn in these_buttons:
                    btn.value = False
            else:
                for btn in these_buttons:
                    btn.value = True
            _refresh_plsda()
        toggle_all_button.on_click(on_toggle_all_clicked)
        
        # Arrange the feature buttons in a grid
        grid = widgets.GridBox(
            children=buttons,
            layout=widgets.Layout(
                grid_template_columns="repeat(8, 160px)",
                grid_gap="10px 10px"
            )
        )
        header = widgets.HBox([widgets.HTML(value=f"<b>{group_name}</b>"), toggle_all_button])
        group_box = widgets.VBox([header, grid])
        group_widgets.append(group_box)
    
    # Global select/deselect buttons for all features.
    select_all_button = widgets.Button(
        description="Select All Features",
        button_style='success',
        layout=widgets.Layout(width='180px', height='40px')
    )
    deselect_all_button = widgets.Button(
        description="Deselect All Features",
        button_style='warning',
        layout=widgets.Layout(width='180px', height='40px')
    )
    
    def select_all_features(b):
        for btn in all_feature_buttons:
            btn.value = True
        _refresh_plsda()
    
    def deselect_all_features(b):
        for btn in all_feature_buttons:
            btn.value = False
        _refresh_plsda()
    
    select_all_button.on_click(select_all_features)
    deselect_all_button.on_click(deselect_all_features)
    
    # Output widget for plot, contributions table and evaluation metrics
    output_plot = widgets.Output()
    
    # Define a helper function to re-run PLS-DA whenever toggles change.
    def _refresh_plsda():
        selected = [btn.description for btn in all_feature_buttons if btn.value]
        update_plsda_func(selected, output_plot)
    
    # Observe each toggle so that changes trigger a refresh.
    for btn in all_feature_buttons:
        btn.observe(lambda change, b=btn: _refresh_plsda() 
                    if change['name'] == 'value' and change['new'] != change['old'] else None, names='value')
    
    bulk_buttons = widgets.HBox([select_all_button, deselect_all_button])
    full_layout = widgets.VBox([bulk_buttons] + group_widgets + [output_plot])
    
    # Trigger an initial update so that the plot is displayed.
    _refresh_plsda()
    return full_layout

# ====================================================
# Main PLS-DA Application Function
# ====================================================
def run_plsda_analysis(filepath, target_column='Class'):
    """
    Orchestrates the PLS-DA analysis:
      1. Loads data from an Excel file.
      2. Builds a UI for manual feature selection with logical groups.
      3. Performs PLS-DA on selected features.
      4. Displays a ggplot-styled scatter plot with explained variance percentages.
      5. Displays a top 10 contributions table and evaluation metrics side-by-side.
    """
    data_clean, numeric_data, target_data = load_and_prepare_data(filepath, target_column)
    if numeric_data is None:
        print("No numeric data found or file could not be loaded.")
        return
    
    # Container to hold the latest PLS-DA dataframe (for potential future use)
    plsda_df_container = {}
    
    def update_plsda(selected_features, output_widget):
        with output_widget:
            clear_output()
            if len(selected_features) < 2:
                print("Please select at least two features for PLS-DA.")
                return
            
            # Perform PLS-DA analysis
            pls, scores, pls1_percent, pls2_percent = perform_plsda(numeric_data, selected_features, target_data)
            pls_df = pd.DataFrame(scores, columns=['Component1', 'Component2'])
            # Add the target class label (resetting index to align with scores)
            pls_df['Class'] = target_data.reset_index(drop=True)
            
            # Create and display the scatter plot with custom colors and explained variance in labels.
            fig = create_scatter_plot(pls_df, pls1_percent, pls2_percent)
            fig.show()
            
            # -----------------------------
            # Compute Contributions
            # -----------------------------
            contrib_df = pd.DataFrame(pls.x_weights_, index=selected_features, columns=['PLS1', 'PLS2'])
            top_features_pls1 = contrib_df['PLS1'].abs().sort_values(ascending=False).head(10).index
            top_features_pls2 = contrib_df['PLS2'].abs().sort_values(ascending=False).head(10).index
            top_values_pls1 = [contrib_df.loc[feat, 'PLS1'] for feat in top_features_pls1]
            top_values_pls2 = [contrib_df.loc[feat, 'PLS2'] for feat in top_features_pls2]
            top_contrib_df = pd.DataFrame({
                'PLS1': [f"{feat}: {val:.3f}" for feat, val in zip(top_features_pls1, top_values_pls1)],
                'PLS2': [f"{feat}: {val:.3f}" for feat, val in zip(top_features_pls2, top_values_pls2)]
            }).reset_index(drop=True)
            top_contrib_df.index += 1
            
            # -----------------------------
            # Compute Evaluation Metrics
            # -----------------------------
            # Recompute scaled features for prediction (to avoid re-fitting)
            X = numeric_data[selected_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            y_dummies = pd.get_dummies(target_data)
            y_pred_continuous = pls.predict(X_scaled)
            predicted_indices = np.argmax(y_pred_continuous, axis=1)
            predicted_labels = [y_dummies.columns[i] for i in predicted_indices]
            true_labels = target_data.reset_index(drop=True)
            
            acc = accuracy_score(true_labels, predicted_labels)
            f1 = f1_score(true_labels, predicted_labels, average='macro')
            sil = silhouette_score_latent(scores, true_labels)
            fdr = fisher_ratio(scores, true_labels)
            distances = pairwise_class_distances(scores, true_labels)
            
            # Prepare outputs for contributions and metrics side-by-side
            contrib_out = widgets.Output()
            metrics_out = widgets.Output()
            
            with contrib_out:
                print("Top 10 Feature Contributions:")
                display(top_contrib_df)
            
            with metrics_out:
                print("Evaluation Metrics:\n")
                print(f"Accuracy: {acc:.3f}")
                print(f"F1 Score (macro): {f1:.3f}\n")
                print(f"Silhouette Score: {sil:.3f}\n")
                print(f"Fisher Discriminant Ratio: {fdr:.3f}\n")
                print("Pairwise Class Distances:")
                for pair, dist in distances.items():
                    print(f"  {pair}: {dist:.3f}")
            
            metrics_out.layout = widgets.Layout(width='400px', margin='0 0 0 20px')
            display(widgets.HBox([contrib_out, metrics_out]))
            
            # Save results in container if needed later
            plsda_df_container['pls_df'] = pls_df
    
    # Build the UI layout with feature grouping.
    ui_layout = build_plsda_ui(
        all_features=list(numeric_data.columns),
        update_plsda_func=update_plsda,
        plsda_df_container=plsda_df_container
    )
    
    display(ui_layout)