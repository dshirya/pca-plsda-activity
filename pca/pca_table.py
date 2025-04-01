import pandas as pd
from IPython.display import display, clear_output

from .data_loader import load_and_prepare_data
from .pca import perform_pca, get_element_group_mapping, create_scatter_plot
from .ui_widgets import build_pca_ui

def run_pca_analysis(filepath):
    """
    Main orchestration function: 
      1. Loads data from 'filepath'
      2. Builds containers to hold latest PCA results
      3. Defines the PCA-update callback
      4. Calls 'build_pca_ui' to create all UI 
      5. Displays everything
    """

    data_clean, numeric_data, symbol_data = load_and_prepare_data(filepath)
    if numeric_data is None:
        print("No numeric data found or file could not be loaded.")
        return

    pca_df_container = {}
    contrib_df_container = {}

    def update_pca(selected_features, output_plot_widget):
        with output_plot_widget:
            clear_output()
            if len(selected_features) < 2:
                print("Please select at least two features for PCA.")
                return

            pca, principal_components = perform_pca(numeric_data, selected_features)
            pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

            if symbol_data is not None:
                pca_df['Symbol'] = symbol_data.reset_index(drop=True)
            else:
                pca_df['Symbol'] = pca_df.index.astype(str)

            pc1_percent = pca.explained_variance_ratio_[0] * 100
            pc2_percent = pca.explained_variance_ratio_[1] * 100

            element_to_group = get_element_group_mapping()
            pca_df["group"] = pca_df["Symbol"].apply(lambda x: element_to_group.get(x, "Other"))

            group_colors = {
                "alkali_metals": "blue",
                "alkaline_earth_metals": "turquoise",
                "transition_metals": "palegreen",
                "lanthanides": "yellow",
                "actinides": "goldenrod",
                "metalloids": "orange",
                "non_metals": "orangered",
                "halogens": "red",
                "noble_gases": "skyblue",
                "post_transition_metals": "darkgreen",
                "Other": "grey"
            }

            fig = create_scatter_plot(pca_df, group_colors, pc1_percent, pc2_percent)
            fig.show()

            contrib_df = pd.DataFrame(
                pca.components_.T,
                index=selected_features,
                columns=['PC1', 'PC2']
            )

            top_features_pc1 = contrib_df['PC1'].abs().sort_values(ascending=False).head(10).index
            top_features_pc2 = contrib_df['PC2'].abs().sort_values(ascending=False).head(10).index

            top_values_pc1 = [contrib_df.loc[f, 'PC1'] for f in top_features_pc1]
            top_values_pc2 = [contrib_df.loc[f, 'PC2'] for f in top_features_pc2]

            top_contrib_df = pd.DataFrame({
                'PC1': [f"{feat}: {val:.3f}" for feat, val in zip(top_features_pc1, top_values_pc1)],
                'PC2': [f"{feat}: {val:.3f}" for feat, val in zip(top_features_pc2, top_values_pc2)]
            }).reset_index(drop=True)
            top_contrib_df.index += 1

            print("\n\033[1mTop 10 Feature Contributions:\033[0m")
            display(top_contrib_df)

            pca_df_container['pca_df'] = pca_df
            contrib_df_container['contrib_df'] = contrib_df

    # Build the UI
    ui_layout = build_pca_ui(
        all_features=list(numeric_data.columns),
        update_pca_func=update_pca,
        pca_df_container=pca_df_container,
        contrib_df_container=contrib_df_container
    )

    display(ui_layout)
   