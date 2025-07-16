import os
import numpy as np

# App configuration
PORT = int(os.environ.get("PORT", 8080))
RANDOM_SEED = 42

# Set random seed
np.random.seed(RANDOM_SEED)

# Data file paths
DATA_PATHS = {
    "features_binary": "data/features-binary.xlsx",
    "elemental_properties": "data/elemental-property-list.xlsx", 
    "pauling_data": "data/pauling-data.xlsx"
}

# Column configuration
LABEL_COLUMN = "Class"

# Default feature selections
DEFAULT_PLS_FEATURES = {
    "Pauling_radius_CN12_A/B",
    "atomic_weight_A+B_weighted"
}

DEFAULT_PCA_FEATURES = {
    "Period",
    "Group"
}

# PLS-DA algorithm parameters
PLSDA_PARAMS = {
    "n_components": 2,
    "plateau_steps": 10,
    "min_features": 2,
    "scoring": "accuracy"
}

# UI configuration
UI_CONFIG = {
    "sidebar_width": 875,
    "plot_width": 800,
    "plot_height": 800,
    "cluster_plot_height": 830,
    "eval_plot_height": 300,
    "scatter_plot_width": 600,
    "scatter_plot_height": 550
}

# Color schemes
COLOR_SCHEMES = {
    "class_colors": {
        0: "#c3121e", 1: "#0348a1", 2: "#ffb01c", 3: "#027608",
        4: "#1dace6", 5: "#9c5300", 6: "#9966cc", 7: "#ff4500"
    },
    "structure_colors": {
        "CsCl": "#c3121e",
        "NaCl": "#0348a1", 
        "ZnS": "#ffb01c"
    },
    "element_group_colors": {
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
} 