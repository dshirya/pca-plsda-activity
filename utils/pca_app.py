import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

from .data_loader import load_and_prepare_data, get_unique_filename
from .ui_widgets import (
    create_toggle_buttons, 
    create_bulk_buttons,
    create_save_buttons  # <-- newly imported
)
from .pca import perform_pca, get_element_group_mapping, create_scatter_plot

def run_pca_analysis(filepath):
    data_clean, numeric_data, symbol_data = load_and_prepare_data(filepath)
    if numeric_data is None:
        return

    current_numeric_data = numeric_data
    current_symbol_data = symbol_data.reset_index(drop=True) if symbol_data is not None else None
    features = list(current_numeric_data.columns)

    # We'll store the most recent PCA results so the Save buttons can access them
    pca_df_container = {}
    contrib_df_container = {}

    def update_pca(change=None):
        selected_features = [
            btn.description for btn in all_feature_buttons if btn.value
        ]
        with output_plot:
            clear_output()
            if len(selected_features) < 2:
                print("Please select at least two features for PCA.")
                return

            pca, principal_components = perform_pca(current_numeric_data, selected_features)
            pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
            if current_symbol_data is not None:
                pca_df['Symbol'] = current_symbol_data.reset_index(drop=True)
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
            
            top_features_pc1 = contrib_df['PC1'].abs().sort_values(ascending=False).head(10).index.tolist()
            top_features_pc2 = contrib_df['PC2'].abs().sort_values(ascending=False).head(10).index.tolist()
            
            top_values_pc1 = [contrib_df.loc[f, 'PC1'] for f in top_features_pc1]
            top_values_pc2 = [contrib_df.loc[f, 'PC2'] for f in top_features_pc2]
            
            top_contrib_df = pd.DataFrame({
                'PC1': [f"{feat}: {val:.3f}" for feat, val in zip(top_features_pc1, top_values_pc1)],
                'PC2': [f"{feat}: {val:.3f}" for feat, val in zip(top_features_pc2, top_values_pc2)]
            }).reset_index(drop=True)
            top_contrib_df.index += 1
            
            print("\nTop 10 Feature Contributions:")
            display(top_contrib_df)
            
            pca_df_container['pca_df'] = pca_df
            contrib_df_container['contrib_df'] = contrib_df

    # Define the callbacks for saving
    def on_save_coordinates_clicked(b):
        if 'pca_df' not in pca_df_container:
            print("No PCA coordinates to save. Please run PCA first.")
            return
        pca_df = pca_df_container['pca_df']
        outfile = get_unique_filename("outputs/elements_pca_coordinates.xlsx")
        pca_df.to_excel(outfile, index=False)
        print(f"Coordinates saved to: {outfile}")

    def on_save_contributions_clicked(b):
        if 'contrib_df' not in contrib_df_container:
            print("No PCA contributions to save. Please run PCA first.")
            return
        contrib_df = contrib_df_container['contrib_df']
        outfile = get_unique_filename("outputs/pca_properties_contributions.xlsx")
        contrib_df_out = contrib_df.reset_index().rename(columns={'index': 'Property'})
        contrib_df_out.to_excel(outfile, index=False)
        print(f"Contributions saved to: {outfile}")

    # Create category widgets (feature toggles)
    category_widgets = create_toggle_buttons(features, update_pca)

    # Extract all individual ToggleButtons
    all_feature_buttons = []
    for cat_widget in category_widgets:
        if len(cat_widget.children) > 1:
            grid = cat_widget.children[1]
            if isinstance(grid, widgets.GridBox):
                for child in grid.children:
                    if isinstance(child, widgets.ToggleButton):
                        all_feature_buttons.append(child)

    # The PCA plot area
    output_plot = widgets.Output()

    # Observe toggles
    for btn in all_feature_buttons:
        btn.observe(update_pca, names='value')

    # Create “Select All / Deselect All” buttons
    select_all_button, deselect_all_button = create_bulk_buttons(all_feature_buttons, update_pca)
    
    # Create the “Save” buttons (just definitions – callback is external)
    save_coordinates_button, save_contributions_button = create_save_buttons(
        on_save_coordinates_clicked, 
        on_save_contributions_clicked
    )
    
    # Create a spacer widget. For example:
    spacer = widgets.Box(layout=widgets.Layout(width="150px"))  # 30px gap   
    
    # Put all four buttons in the same row
    bulk_and_save_buttons = widgets.HBox([
        select_all_button, 
        deselect_all_button, 
        spacer,
        save_coordinates_button, 
        save_contributions_button
    ])

    # Initial run (optional)
    update_pca()

    # Display
    display(bulk_and_save_buttons, *category_widgets, output_plot)