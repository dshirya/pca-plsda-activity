import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, clear_output

def load_and_prepare_data(filepath):
    """
    Loads an Excel file, drops columns with missing values,
    and selects numeric columns (dropping any rows with missing values).
    Optionally returns the 'Symbol' column if available.
    """
    try:
        data = pd.read_excel(filepath)
        data_clean = data.dropna(axis=1)
        numeric_data = data_clean.select_dtypes(include=['float64', 'int64']).copy()
        numeric_data.dropna(inplace=True)
        symbol_data = data_clean.loc[numeric_data.index, 'Symbol'] if 'Symbol' in data_clean.columns else None
        return data_clean, numeric_data, symbol_data
    except Exception as e:
        print("Error loading file:", e)
        return None, None, None

def create_toggle_buttons(features, run_pca_callback):
    """
    Creates a categorized list of ToggleButton widgets for each feature,
    arranged in a structured grid with an extra "Toggle All" button for each group.
    """
    categories = {
        "Basic Atomic Properties": [
            'Atomic weight', 'Atomic number', 'Period', 'Group', 'Families'
        ],
        "Electronic Structure": [
            'Mendeleev number', 'quantum  number l', 'valence s', 'valence p', 'valence d', 'valence f', 
            'unfilled s', 'unfilled p', 'unfilled d', 'unfilled f', 'no. of  valence  electrons', 
            'outer shell electrons', 'Gilman no. of valence electrons', 'Metallic  valence', 'Zeff', '1st Bohr radius (a0)'
        ],
        "Electronegativity & Electron Affinity": [
            'Ionization energy (eV)', 'Electron affinity (ev)', 'Pauling EN', 'Martynov Batsanov EN',
            'Mulliken EN', 'Allred EN', 'Allred Rockow EN', 'Nagle EN', 'Ghosh EN'
        ],
        "Atomic & Ionic Radii": [
            'Atomic radius calculated', 'Covalent radius', 'Ionic radius', 'Effective ionic radius',
            'Miracle radius', 'van der Waals radius', 'Zunger radii sum', 'Crystal radius', 
            'Covalent CSD radius', 'Slater radius', 'Orbital radius', 'polarizability, A^3'
        ],
        "Thermal & Physical Properties": [
            'Melting point, K', 'Boiling point, K', 'Density,  g/mL', 'Specific heat, J/g K', 
            'Heat of fusion,  kJ/mol', 'Heat of vaporization,  kJ/mol', 'Heat of atomization,  kJ/mol', 
            'Thermal conductivity, W/m K', 'Cohesive  energy', 'Bulk modulus, GPa'
        ],
        "DFT LSD & LDA Properties": [
            'DFT LDA Etot', 'DFT LDA Ekin', 'DFT LDA Ecoul', 'DFT LDA Eenuc', 'DFT LDA Exc',
            'DFT LSD Etot', 'DFT LSD Ekin', 'DFT LSD Ecoul', 'DFT LSD Eenuc', 'DFT LSD Exc'
        ],
        "DFT RLDA & ScRLDA Properties": [
            'DFT RLDA Etot', 'DFT RLDA Ekin', 'DFT RLDA Ecoul', 'DFT RLDA Eenuc', 'DFT RLDA Exc',
            'DFT ScRLDA Etot', 'DFT ScRLDA Ekin', 'DFT ScRLDA Ecoul', 'DFT ScRLDA Eenuc', 'DFT ScRLDA Exc'
        ]
    }
    
    category_widgets = []
    for category, feature_list in categories.items():
        # Create toggle buttons only for features available in the dataset
        buttons = [
            widgets.ToggleButton(
                value=True,
                description=feat,
                tooltip=f"Toggle feature: {feat}",
                layout=widgets.Layout(width='190px', height='30px')
            ) 
            for feat in feature_list if feat in features
        ]
        if not buttons:
            continue  # Skip categories with no matching features
        
        # Create a "Toggle All" button for the group
        toggle_all_button = widgets.Button(
            description="Toggle",
            button_style='primary',
            layout=widgets.Layout(width='60px', height='25px', margin='5px')
        )
        def on_toggle_all(b, buttons=buttons):
            # If all buttons are selected, deselect them; otherwise, select all.
            if all(btn.value for btn in buttons):
                for btn in buttons:
                    btn.value = False
            else:
                for btn in buttons:
                    btn.value = True
            run_pca_callback()
        toggle_all_button.on_click(on_toggle_all)
        
        # Build header with the category name and the group toggle button.
        category_label = widgets.HTML(value=f"<b>{category}</b>")
        header = widgets.HBox([category_label, toggle_all_button])
        
        grid = widgets.GridBox(
            children=buttons,
            layout=widgets.Layout(
                grid_template_columns="repeat(10, 200px)",
                grid_gap="10px 10px"
            )
        )
        # Combine header and grid in a vertical box.
        category_box = widgets.VBox([header, grid])
        category_widgets.append(category_box)
    
    return category_widgets

def create_bulk_buttons(toggle_buttons, run_pca_callback):
    """
    Creates global bulk action buttons for feature toggling.
    Only the "Select All" and "Deselect All" buttons remain.
    """
    select_all_button = widgets.Button(
        description="Select All Features",
        button_style='success',
        layout=widgets.Layout(width='200px', height='40px')
    )
    deselect_all_button = widgets.Button(
        description="Deselect All Features",
        button_style='warning',
        layout=widgets.Layout(width='200px', height='40px')
    )
    
    def select_all_features(b):
        for btn in toggle_buttons:
            btn.value = True
        run_pca_callback()
        
    def deselect_all_features(b):
        for btn in toggle_buttons:
            btn.value = False
        run_pca_callback()
    
    select_all_button.on_click(select_all_features)
    deselect_all_button.on_click(deselect_all_features)
    
    return select_all_button, deselect_all_button

def perform_pca(data, selected_features):
    """
    Standardizes the selected data and performs PCA (reducing to 2 components).
    """
    data_subset = data[selected_features].copy()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_subset)
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(scaled_data)
    return pca, principal_components

def get_element_group_mapping():
    """
    Returns a dictionary mapping element symbols to their groups.
    """
    elements_by_group = {
        "alkali_metals": ["Li", "Na", "K", "Rb", "Cs", "Fr"],
        "alkaline_earth_metals": ["Be", "Mg", "Ca", "Sr", "Ba", "Ra"],
        "transition_metals": [
            "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
            "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"
        ],
        "lanthanides": [
            "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", 
            "Ho", "Er", "Tm", "Yb", "Lu"
        ],
        "actinides": [
            "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", 
            "Es", "Fm", "Md", "No", "Lr"
        ],
        "metalloids": ["B", "Si", "Ge", "As", "Sb", "Te", "Po"],
        "non_metals": ["H", "C", "N", "O", "P", "S", "Se"],
        "halogens": ["F", "Cl", "Br", "I", "At", "Ts"],
        "noble_gases": ["He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"],
        "post_transition_metals": ["Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Nh", "Fl", "Mc", "Lv"]
    }
    element_to_group = {}
    for group, elems in elements_by_group.items():
        for elem in elems:
            element_to_group[elem] = group
    return element_to_group

def save_pca_results(pca_df, pca_components, selected_features):
    """
    Saves the PCA coordinates and feature contributions to Excel files.
    """
    pca_coordinates_file = "elements_pca_coordinates.xlsx"
    pca_df.to_excel(pca_coordinates_file, index=False)
    
    pca_contribution = pd.DataFrame(
        pca_components,
        columns=selected_features,
        index=['PC1', 'PC2']
    ).T.reset_index()
    pca_contribution.columns = ['Property', 'PC1', 'PC2']
    contribution_file = "pca_properties_contributions.xlsx"
    pca_contribution.to_excel(contribution_file, index=False)
    
    return pca_coordinates_file, contribution_file

def create_scatter_plot(pca_df, group_colors, pc1_percent, pc2_percent):
    """
    Creates and returns a Plotly scatter plot of the PCA results with
    axis labels showing explained variance percentages.
    """
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        text='Symbol',
        labels={
            "PC1": f"PC 1 ({pc1_percent:.1f}%)",
            "PC2": f"PC 2 ({pc2_percent:.1f}%)"
        },
        template="ggplot2",
        color="group",
        color_discrete_map=group_colors
    )
    fig.update_traces(marker=dict(size=26, symbol="circle", opacity=0.6))                                               # markers 
    fig.update_layout(
        width=1200,
        height=1200,
        xaxis=dict(scaleanchor='y', scaleratio=1),
        yaxis=dict(scaleanchor='x', scaleratio=1),
        showlegend=False,
        font=dict(size=18)
    )
    return fig

def run_pca_analysis(filepath):
    """
    Main function to run the PCA analysis. Loads data from the given file,
    sets up the interactive feature selection, performs PCA, and displays
    the Plotly scatter plot along with a table of the top 10 contributions.
    """
    data_clean, numeric_data, symbol_data = load_and_prepare_data(filepath)
    if numeric_data is None:
        return
    current_numeric_data = numeric_data
    current_symbol_data = symbol_data.reset_index(drop=True) if symbol_data is not None else None
    
    #print("Available numeric columns:", list(current_numeric_data.columns))
    
    features = list(current_numeric_data.columns)
    
    # Define the callback to update the PCA plot.
    def update_pca(change=None):
        # Gather selected features from all individual toggle buttons.
        selected_features = [
            btn.description for btn in all_feature_buttons if btn.value
        ]
        with output_plot:
            clear_output()
            if len(selected_features) < 2:
                print("Please select at least two features for PCA.")
                return

            # Perform PCA
            pca, principal_components = perform_pca(current_numeric_data, selected_features)
            pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
            if current_symbol_data is not None:
                pca_df['Symbol'] = current_symbol_data.reset_index(drop=True)
            else:
                pca_df['Symbol'] = pca_df.index.astype(str)
            
            # Calculate explained variance percentages
            pc1_percent = pca.explained_variance_ratio_[0] * 100
            pc2_percent = pca.explained_variance_ratio_[1] * 100
            
            # Map element symbols to groups
            element_to_group = get_element_group_mapping()
            pca_df["group"] = pca_df["Symbol"].apply(lambda x: element_to_group.get(x, "Other"))
            
            # Define group colors mapping
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
            
            # Save PCA results to files
            pca_coordinates_file, contribution_file = save_pca_results(pca_df, pca.components_, selected_features)
            # print("PCA coordinates saved to", pca_coordinates_file)
            # print("PCA contributions saved to", contribution_file)
            
            # Create and display the scatter plot
            fig = create_scatter_plot(pca_df, group_colors, pc1_percent, pc2_percent)
            fig.show()
            
            # Create a DataFrame for the contributions (loadings)
            contrib_df = pd.DataFrame(pca.components_.T, index=selected_features, columns=['PC1', 'PC2'])
            
            # For each principal component, select the top 10 features by absolute contribution
            top_features_pc1 = contrib_df['PC1'].abs().sort_values(ascending=False).head(10).index.tolist()
            top_features_pc2 = contrib_df['PC2'].abs().sort_values(ascending=False).head(10).index.tolist()
            top_values_pc1 = [contrib_df.loc[f, 'PC1'] for f in top_features_pc1]
            top_values_pc2 = [contrib_df.loc[f, 'PC2'] for f in top_features_pc2]
            
            # Build a rotated DataFrame that shows the top contributions for PC1 and PC2
            top_contrib_df = pd.DataFrame({
                'PC1': [f"{feat}: {val:.3f}" for feat, val in zip(top_features_pc1, top_values_pc1)],
                'PC2': [f"{feat}: {val:.3f}" for feat, val in zip(top_features_pc2, top_values_pc2)]
            })
            
            print("\nTop 10 Feature Contributions:")
            display(top_contrib_df)
    
    # Create category widgets (with per-group toggle buttons)
    category_widgets = create_toggle_buttons(features, update_pca)
    
    # Extract all individual feature toggle buttons from each category grid.
    all_feature_buttons = []
    for cat_widget in category_widgets:
        # Each category widget is a VBox with [header, grid]
        if len(cat_widget.children) > 1:
            grid = cat_widget.children[1]
            if isinstance(grid, widgets.GridBox):
                all_feature_buttons.extend([btn for btn in grid.children if isinstance(btn, widgets.ToggleButton)])
    
    # Output widget for the PCA plot and messages.
    output_plot = widgets.Output()
    
    # Attach observer to each individual feature toggle button.
    for btn in all_feature_buttons:
        btn.observe(update_pca, names='value')
    
    # Create global bulk buttons for selecting/deselecting all features.
    select_all_button, deselect_all_button = create_bulk_buttons(all_feature_buttons, update_pca)
    bulk_buttons_row = widgets.HBox([select_all_button, deselect_all_button])
    
    # Initial run of PCA plot.
    update_pca()
    
    # Display the global bulk buttons, category widgets, and PCA output.
    display(bulk_buttons_row, *category_widgets, output_plot)