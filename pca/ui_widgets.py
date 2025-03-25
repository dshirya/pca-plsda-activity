import ipywidgets as widgets

# For unique filenames when saving
from .data_loader import get_unique_filename

def build_pca_ui(all_features, update_pca_func,
                 pca_df_container, contrib_df_container):
    """
    Builds and returns a complete UI layout for controlling PCA.

    Parameters
    ----------
    all_features : list of str
        All numeric columns in your dataset.

    update_pca_func : function
        The callback that actually performs PCA. Must accept (selected_features, output_widget).

    pca_df_container : dict
        A dict to hold the latest PCA dataframe under 'pca_df'.

    contrib_df_container : dict
        A dict to hold the latest contributions dataframe under 'contrib_df'.

    Returns
    -------
    An ipywidgets layout (e.g. VBox) that you can display in your notebook.
    """

    # ========== 1) CREATE CATEGORY TOGGLES ==========
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

    for category, feature_list in categories.items():
        # Filter out features not in your dataset
        buttons = [
            widgets.ToggleButton(
                value=True,  # default to True
                description=feat,
                tooltip=f"Toggle feature: {feat}",
                layout=widgets.Layout(width='190px', height='30px')
            )
            for feat in feature_list if feat in all_features
        ]
        if not buttons:
            continue
        
        # "Toggle" button for the entire category
        toggle_all_button = widgets.Button(
            description="Toggle",
            button_style='primary',
            layout=widgets.Layout(width='60px', height='25px', margin='5px')
        )
        def on_toggle_all_clicked(b, these_buttons=buttons):
            if all(btn.value for btn in these_buttons):
                for btn in these_buttons:
                    btn.value = False
            else:
                for btn in these_buttons:
                    btn.value = True
            _refresh_pca()
        toggle_all_button.on_click(on_toggle_all_clicked)

        grid = widgets.GridBox(
            children=buttons,
            layout=widgets.Layout(
                grid_template_columns="repeat(10, 200px)",
                grid_gap="10px 10px"
            )
        )

        category_label = widgets.HTML(value=f"<b>{category}</b>")
        header = widgets.HBox([category_label, toggle_all_button])
        category_box = widgets.VBox([header, grid])
        category_widgets.append(category_box)

        # Collect these toggles
        all_feature_buttons.extend(buttons)

    # ========== 2) SELECT ALL / DESELECT ALL ==========
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
        _refresh_pca()

    def deselect_all_features(b):
        for btn in all_feature_buttons:
            btn.value = False
        _refresh_pca()

    select_all_button.on_click(select_all_features)
    deselect_all_button.on_click(deselect_all_features)

    # ========== 3) SAVE BUTTONS ==========
    save_coordinates_button = widgets.Button(
        description="Save Coordinates",
        button_style='info',
        layout=widgets.Layout(width='150px', height='40px')
    )
    save_contributions_button = widgets.Button(
        description="Save Contributions",
        button_style='info',
        layout=widgets.Layout(width='150px', height='40px')
    )

    def on_save_coordinates_clicked(b):
        if 'pca_df' not in pca_df_container:
            print("No PCA coordinates to save. Please run PCA first.")
            return
        pca_df = pca_df_container['pca_df']

        # Make a copy so we don't alter the original in the container
        pca_to_save = pca_df.copy()

        # Rename PC1 -> x, PC2 -> y
        pca_to_save.rename(columns={'PC1': 'x', 'PC2': 'y'}, inplace=True)

        outfile = get_unique_filename("outputs/elements_pca_coordinates.xlsx")
        pca_to_save.to_excel(outfile, index=False)
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

    save_coordinates_button.on_click(on_save_coordinates_clicked)
    save_contributions_button.on_click(on_save_contributions_clicked)

    # ========== 4) PCA OUTPUT WIDGET ==========
    output_plot = widgets.Output()

    # Helper to re-run PCA whenever toggles change
    def _refresh_pca():
        selected = [btn.description for btn in all_feature_buttons if btn.value]
        update_pca_func(selected, output_plot)

    # Observe each toggle so changes trigger new PCA
    for btn in all_feature_buttons:
        def on_toggle_change(change, b=btn):
            if change['name'] == 'value' and change['new'] != change['old']:
                _refresh_pca()
        btn.observe(on_toggle_change, names='value')

    # ========== 5) FINAL LAYOUT ==========
    spacer = widgets.Box(layout=widgets.Layout(width="150px"))
    bulk_and_save_buttons = widgets.HBox([
        select_all_button,
        deselect_all_button,
        spacer,
        save_coordinates_button,
        save_contributions_button
    ])

    full_layout = widgets.VBox([
        bulk_and_save_buttons,
        *category_widgets,
        output_plot
    ])

    # ===== TRIGGER INITIAL PCA RUN =====
    # so we see the plot immediately
    _refresh_pca()

    return full_layout