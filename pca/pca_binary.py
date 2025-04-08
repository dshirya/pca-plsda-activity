import os
import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca_analysis_structures(
    filepath: str,
    path_data: pd.DataFrame,
    structure_colors: dict = None,
    structure_markers: dict = None
):
    """
    An interactive PCA analysis that also plots lines and midpoints for each binary
    'Formula' in 'structures_df', colored (and optionally shaped) by 'Structure type'.
    ...
    """
    # ------------------------
    # 1) LOAD & CLEAN DATA
    # ------------------------
    try:
        data = pd.read_excel(filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    structures_df = pd.read_excel(path_data)
    data_clean = data.dropna(axis=1)  # drop columns with all-NaN
    numeric_data = data_clean.select_dtypes(include=['float64', 'int64']).copy()
    numeric_data.dropna(inplace=True)

    if 'Symbol' not in data_clean.columns:
        print("No 'Symbol' column found in the dataset. Cannot proceed.")
        return

    symbol_data = data_clean.loc[numeric_data.index, 'Symbol'].reset_index(drop=True)

    # The default feature set is all numeric columns
    all_features = list(numeric_data.columns)

    # Container to store PCA results so we can "Save Coordinates" etc.
    pca_df_container = {}
    contrib_df_container = {}

    # ------------------------
    # 1.1) SET DEFAULT COLORS IF NONE PROVIDED
    # ------------------------
    if structure_colors is None:
        default_colors = [
            "#c3121e",  # Sangre
            "#0348a1",  # Neptune
            "#ffb01c",  # Pumpkin
            "#027608",  # Clover
            "#1dace6",  # Cerulean
            "#9c5300",  # Cocoa
            "#9966cc",  # Amethyst
            "#ff4500",  # Orange Red
        ]
        unique_types = structures_df["Structure type"].unique()
        structure_colors = {stype: default_colors[i % len(default_colors)]
                            for i, stype in enumerate(unique_types)}

    # Optionally, you can also set default markers if not provided.
    if structure_markers is None:
        # For example, default to 'circle' for all types:
        unique_types = structures_df["Structure type"].unique()
        structure_markers = {stype: 'circle' for stype in unique_types}


    # ------------------------
    # 2) BUILD THE UI
    # ------------------------

    # ========== 2.1) Feature Toggle Buttons by Category ==========
    categories = {
        "Basic Atomic Properties": [
            'Atomic weight', 'Atomic number', 'Period', 'Group', 'Families'
        ],
        "Electronic Structure": [
            'Mendeleev number', 'quantum  number l', 'valence s', 'valence p',
            'valence d', 'valence f', 'unfilled s', 'unfilled p', 'unfilled d',
            'unfilled f', 'no. of  valence  electrons','outer shell electrons',
            'Gilman no. of valence electrons', 'Metallic  valence', 'Zeff', '1st Bohr radius (a0)'
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
    all_feature_buttons = []

    def on_toggle_all_clicked(b, these_buttons):
        """Toggle all feature buttons in a category."""
        if all(btn.value for btn in these_buttons):
            for btn in these_buttons:
                btn.value = False
        else:
            for btn in these_buttons:
                btn.value = True
        _update_pca()

    for category, feat_list in categories.items():
        cat_features = [f for f in feat_list if f in all_features]
        if not cat_features:
            continue
        buttons = []
        for feat in cat_features:
            tb = widgets.ToggleButton(
                value=True,  # default on
                description=feat,
                tooltip=f"Toggle feature: {feat}",
                layout=widgets.Layout(width='190px', height='30px')
            )
            buttons.append(tb)
            all_feature_buttons.append(tb)

        toggle_all_button = widgets.Button(
            description="Toggle",
            button_style='primary',
            layout=widgets.Layout(width='100px', height='25px', margin='5px')
        )
        toggle_all_button.on_click(lambda b, these=buttons: on_toggle_all_clicked(b, these))

        cat_label = widgets.HTML(value=f"<b>{category}</b>")
        header = widgets.HBox([cat_label, toggle_all_button])
        grid = widgets.GridBox(
            children=buttons,
            layout=widgets.Layout(
                grid_template_columns="repeat(10, 200px)",
                grid_gap="10px 10px"
            )
        )
        category_widgets.append(widgets.VBox([header, grid]))

    # ========== 2.2) Bulk Buttons: Select All, Deselect All ==========
    select_all_button = widgets.Button(
        description="Select All Features",
        button_style='success',
        layout=widgets.Layout(width='150px', height='40px')
    )
    deselect_all_button = widgets.Button(
        description="Deselect All Features",
        button_style='warning',
        layout=widgets.Layout(width='150px', height='40px')
    )

    def select_all_features(b):
        for btn in all_feature_buttons:
            btn.value = True
        _update_pca()

    def deselect_all_features(b):
        for btn in all_feature_buttons:
            btn.value = False
        _update_pca()

    select_all_button.on_click(select_all_features)
    deselect_all_button.on_click(deselect_all_features)

    # ========== 2.3) Save Buttons ==========
    save_coords_btn = widgets.Button(
        description="Save Coordinates",
        button_style='info',
        layout=widgets.Layout(width='150px', height='40px')
    )
    save_contrib_btn = widgets.Button(
        description="Save Contributions",
        button_style='info',
        layout=widgets.Layout(width='150px', height='40px')
    )

    def get_unique_filename(outfile):
        base, ext = os.path.splitext(outfile)
        counter = 1
        new_outfile = outfile
        while os.path.exists(new_outfile):
            new_outfile = f"{base}_{counter}{ext}"
            counter += 1
        return new_outfile

    def on_save_coords_clicked(b):
        if 'pca_df' not in pca_df_container:
            print("No PCA coordinates to save. Perform PCA first.")
            return
        df = pca_df_container['pca_df'].copy()
        df.rename(columns={'PC1': 'x', 'PC2': 'y'}, inplace=True)

        outfile = get_unique_filename("outputs/elements_pca_coordinates.xlsx")
        df.to_excel(outfile, index=False)
        print(f"Coordinates saved to: {outfile}")

    def on_save_contrib_clicked(b):
        if 'contrib_df' not in contrib_df_container:
            print("No PCA contributions to save. Perform PCA first.")
            return
        cdf = contrib_df_container['contrib_df'].reset_index().rename(columns={'index': 'Property'})
        outfile = get_unique_filename("outputs/pca_properties_contributions.xlsx")
        cdf.to_excel(outfile, index=False)
        print(f"Contributions saved to: {outfile}")

    save_coords_btn.on_click(on_save_coords_clicked)
    save_contrib_btn.on_click(on_save_contrib_clicked)

    # ========== 2.4) Output area for Plot + Messages ==========
    output_plot = widgets.Output()

    # ========== 2.5) Final top row of buttons ==========
    spacer = widgets.Box(layout=widgets.Layout(width='100px'))
    bulk_and_save_buttons = widgets.HBox([
        select_all_button,
        deselect_all_button,
        spacer,
        save_coords_btn,
        save_contrib_btn
    ])

    # ------------------------
    # 3) THE MAIN PCA UPDATE CALLBACK
    # ------------------------
    def _update_pca():
        """
        Gathers selected features, runs PCA, plots the scatter,
        then draws lines for each row in 'structures_df'.
        """
        selected_feats = [tb.description for tb in all_feature_buttons if tb.value]

        with output_plot:
            clear_output()
            if len(selected_feats) < 2:
                print("Please select at least two features for PCA.")
                return

            # Standardize + PCA
            subdata = numeric_data[selected_feats].copy()
            scaled = StandardScaler().fit_transform(subdata)
            pca = PCA(n_components=2, random_state=42)
            principal_components = pca.fit_transform(scaled)

            pca_df = pd.DataFrame(principal_components, columns=['PC1','PC2'])
            pca_df['Symbol'] = symbol_data.reset_index(drop=True)

            pc1_percent = pca.explained_variance_ratio_[0] * 100
            pc2_percent = pca.explained_variance_ratio_[1] * 100

            # Make a base scatter plot
            fig = px.scatter(
                pca_df,
                x='PC1', y='PC2',
                text='Symbol',
                labels={
                    "PC1": f"PC 1 ({pc1_percent:.1f}%)",
                    "PC2": f"PC 2 ({pc2_percent:.1f}%)"
                },
                template="ggplot2"
            )
            fig.update_traces(marker=dict(size=24, opacity=0.3, color='rgb(2, 118, 8)'))

            # Enforce same scale on x & y
            fig.update_layout(
                xaxis=dict(scaleanchor='y', scaleratio=1),
                yaxis=dict(scaleanchor='x', scaleratio=1),
                width=1000,
                height=1000,
                #title="Pauling PCA Analysis",
                font=dict(size=18)
            )
            # Prepare a set to track which structure types we've already
            # shown in the legend, so we only list each type once.
            used_structure_types = set()

            # For each row in 'structures_df', parse the formula, draw line+marker
            for idx, row in structures_df.iterrows():
                formula = row['Formula']
                struct_type = row['Structure type']

                # parse formula (must be binary)
                from .preprocess import parse_formula  # or define parse_formula in same file
                parsed = parse_formula(formula)
                if len(parsed) != 2:
                    # skip if it's not exactly two elements
                    continue
                elements_list = list(parsed.keys())  # e.g. ["Ag","Mg"]
                e1, e2 = elements_list[0], elements_list[1]

                # find their PCA coords
                df1 = pca_df[pca_df['Symbol'] == e1]
                df2 = pca_df[pca_df['Symbol'] == e2]
                if df1.empty or df2.empty:
                    continue

                x1, y1 = df1['PC1'].values[0], df1['PC2'].values[0]
                x2, y2 = df2['PC1'].values[0], df2['PC2'].values[0]

                # midpoint
                midx, midy = (x1+x2)/2.0, (y1+y2)/2.0

                color = structure_colors.get(struct_type, 'black')
                symbol = structure_markers.get(struct_type, 'circle')  

                # If it's the first time we see this structure type, show it in legend
                show_legend = struct_type not in used_structure_types

                # 1) Add the line
                fig.add_scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode='lines',
                    line=dict(color=color, width=0.5),
                    name=struct_type,
                    showlegend=False,  # only the midpoint marker appears in legend
                    hoverinfo='none'
                )

                # 2) Add the midpoint marker
                fig.add_scatter(
                    x=[midx],
                    y=[midy],
                    mode='markers',
                    marker=dict(size=10, color=color, symbol=symbol),
                    name=struct_type,
                    showlegend=show_legend,
                    text=[f"{formula} ({struct_type})"],
                    textposition="top center"
                )

                used_structure_types.add(struct_type)
            fig.show()

            # compute contributions DataFrame
            contrib_df = pd.DataFrame(
                pca.components_.T,
                index=selected_feats,
                columns=['PC1','PC2']
            )

            top_features_pc1 = contrib_df['PC1'].abs().sort_values(ascending=False).head(10).index
            top_features_pc2 = contrib_df['PC2'].abs().sort_values(ascending=False).head(10).index

            top_vals_pc1 = [contrib_df.loc[f, 'PC1'] for f in top_features_pc1]
            top_vals_pc2 = [contrib_df.loc[f, 'PC2'] for f in top_features_pc2]

            top_contrib_df = pd.DataFrame({
                'PC1': [f"{feat}: {val:.3f}" for feat, val in zip(top_features_pc1, top_vals_pc1)],
                'PC2': [f"{feat}: {val:.3f}" for feat, val in zip(top_features_pc2, top_vals_pc2)]
            }).reset_index(drop=True)
            top_contrib_df.index += 1

            print("\n\033[1mTop 10 Feature Contributions:\033[0m")
            display(top_contrib_df)

            # Store in containers so "Save" can see them
            pca_df_container['pca_df'] = pca_df
            contrib_df_container['contrib_df'] = contrib_df

    # ========== WATCH toggles so changes re-run PCA
    for btn in all_feature_buttons:
        def on_toggle_change(change, b=btn):
            if change['name'] == 'value' and change['new'] != change['old']:
                _update_pca()
        btn.observe(on_toggle_change, names='value')

    # ------------------------
    # 4) SHOW THE ENTIRE UI
    # ------------------------
    # Build the final layout
    final_layout = widgets.VBox([
        bulk_and_save_buttons,
        *category_widgets,
        output_plot
    ])
    display(final_layout)

    # Run PCA initially (so we see something before toggles)
    _update_pca()