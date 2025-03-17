import ipywidgets as widgets
from IPython.display import clear_output

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
            continue
        
        toggle_all_button = widgets.Button(
            description="Toggle",
            button_style='primary',
            layout=widgets.Layout(width='60px', height='25px', margin='5px')
        )
        
        def on_toggle_all(b, buttons=buttons):
            if all(btn.value for btn in buttons):
                for btn in buttons:
                    btn.value = False
            else:
                for btn in buttons:
                    btn.value = True
            run_pca_callback()
        
        toggle_all_button.on_click(on_toggle_all)
        
        category_label = widgets.HTML(value=f"<b>{category}</b>")
        header = widgets.HBox([category_label, toggle_all_button])
        
        grid = widgets.GridBox(
            children=buttons,
            layout=widgets.Layout(
                grid_template_columns="repeat(10, 200px)",
                grid_gap="10px 10px"
            )
        )
        category_box = widgets.VBox([header, grid])
        category_widgets.append(category_box)
    
    return category_widgets

def create_bulk_buttons(toggle_buttons, run_pca_callback):
    """
    Creates global bulk action buttons for feature toggling:
    "Select All" and "Deselect All".
    """
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

def create_save_buttons(
    on_save_coordinates_clicked, 
    on_save_contributions_clicked
):
    """
    Creates the two "Save" buttons for PCA coordinates and contributions.
    The callback functions (on_save_coordinates_clicked, 
    on_save_contributions_clicked) should be defined wherever
    run_pca_analysis logic resides, then passed in here.
    """
    
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
    
    # Attach callback functions from outside
    save_coordinates_button.on_click(on_save_coordinates_clicked)
    save_contributions_button.on_click(on_save_contributions_clicked)
    
    return save_coordinates_button, save_contributions_button