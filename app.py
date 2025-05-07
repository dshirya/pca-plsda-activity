import re
import os
import random
import pandas as pd
import numpy as np

from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go

port = int(os.environ.get("PORT", 8080))

# ——————————————
# Load your fixed dataset
# ——————————————
df = pd.read_csv("data/features-binary.csv")

label_col = "Class"  # exact name of your class column
pca_data = "data/elemental-property-list.xlsx"
cluster_data = "data/pauling-data.xlsx"

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


symbol_to_group = {
    el: grp
    for grp, lst in elements_by_group.items()
    for el in lst
}

# ————————————————————————
# Helper to turn any string into a valid Shiny ID
# ————————————————————————
def make_safe_id(name: str) -> str:
    # first give / and + unique replacements
    s = name.replace("/", "_slash_").replace("+", "_plus_")
    # then replace any non-alphanumeric-or-underscore with underscore
    safe = re.sub(r'\W+', '_', s)
    # ensure it doesn’t start with a digit
    if re.match(r'^\d', safe):
        safe = 'f_' + safe
    return safe

# -------------------------------
# Helper: Evaluate a Feature Subset via 5-fold CV
# -------------------------------
def evaluate_subset(X: pd.DataFrame,
                    y: pd.Series,
                    selected_features: list[str],
                    n_components: int = 2,
                    scoring: str = "accuracy") -> float:

    # turn your list into a DataFrame slice
    X_sub = X[selected_features].copy()

    # **DROP any columns whose std == 0**
    zero_var = X_sub.std(axis=0) == 0
    if zero_var.any():
        X_sub = X_sub.loc[:, ~zero_var]
    if X_sub.shape[1] < n_components:
        # not enough features to even form n_components
        return np.nan

    # now scale safely
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    # 2) One‐hot encode y and fit PLS
    Y_dummy = pd.get_dummies(y)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(X_scaled, Y_dummy)

    # 3) Predict on the same data
    y_pred_cont = pls.predict(X_scaled)
    idxs = np.argmax(y_pred_cont, axis=1)
    preds = [Y_dummy.columns[i] for i in idxs]

    # 4) Compute the requested metric
    if scoring == "accuracy":
        return accuracy_score(y, preds)
    else:
        return f1_score(y, preds, average="macro")

# -------------------------------
# Forward Selection on a DataFrame
# -------------------------------
def forward_selection_plsda_df(numeric_data, target_data,
                              max_features=40, n_components=2, scoring='accuracy'):
    features = list(numeric_data.columns)
    remaining = features.copy()
    selected = []
    performance_history = []
    iterations_info = []  # (iteration, selected, score)
    iteration = 0

    # initialize if n_components > 1
    if n_components > 1:
        init = random.sample(remaining, n_components)
        for feat in init:
            remaining.remove(feat)
        selected.extend(init)
        base_score = evaluate_subset(numeric_data, target_data, selected,
                                      n_components=n_components, scoring=scoring)
        performance_history.append(base_score)
        iteration += 1
        iterations_info.append((iteration, selected.copy(), base_score))
    else:
        base_score = -np.inf

    best_score = base_score

    # greedy forward
    while remaining and (max_features is None or len(selected) < max_features):
        best_cand, best_cand_score = None, -np.inf
        for feat in random.sample(remaining, len(remaining)):
            trial = selected + [feat]
            if len(trial) < n_components: 
                continue
            s = evaluate_subset(numeric_data, target_data, trial,
                                 n_components=n_components, scoring=scoring)
            if s > best_cand_score:
                best_cand_score, best_cand = s, feat
        if best_cand is not None and best_cand_score >= best_score:
            remaining.remove(best_cand)
            selected.append(best_cand)
            best_score = best_cand_score
            performance_history.append(best_score)
            iteration += 1
            iterations_info.append((iteration, selected.copy(), best_score))
        else:
            break

    return performance_history, iterations_info

# -------------------------------
# Backward Elimination on a DataFrame
# -------------------------------
def backward_elimination_plsda_df(numeric_data, target_data,
                                 min_features=1, n_components=2, scoring='accuracy'):
    current = list(numeric_data.columns)
    performance_history = []
    iterations_info   = []

    # --- INITIAL ENTRY ---
    best_score = evaluate_subset(
        numeric_data,            # your X-matrix
        target_data,             # your y-vector
        current,                 # the full feature list
        n_components=n_components,
        scoring=scoring
        )
    performance_history.append(best_score)
    iterations_info.append(
        # was (iteration, …); now store number of features
        (len(current), current.copy(), best_score)
    )

    while len(current) > min_features:
        best_after, feat_to_remove = -np.inf, None
        for feat in random.sample(current, len(current)):
            trial = [f for f in current if f != feat]
            s = evaluate_subset(numeric_data, target_data, trial,
                                 n_components=n_components, scoring=scoring)
            if s > best_after:
                best_after, feat_to_remove = s, feat
        if feat_to_remove and best_after >= best_score:
            current.remove(feat_to_remove)
            best_score = best_after
            performance_history.append(best_score)

            # record NEW feature count, not an iteration counter
            iterations_info.append(
                (len(current), current.copy(), best_score)
            )
        else:
            break

    return performance_history, iterations_info

# ——————————————————————————————
# Define your feature groups (display, column) pairs
# ——————————————————————————————
feature_groups_plsda = {
    "index": {
        "label": "Index Features",
        "features": [
            ("A",                 "index_A"),
            ("Index B",                 "index_B"),
            ("Norm A",      "normalized_index_A"),
            ("Norm B",      "normalized_index_B"),
            ("Max",           "largest_index"),
            ("Min",          "smallest_index"),
            ("Average",           "avg_index"),
        ],
    },
    "atomic_weight": {
        "label": "Atomic Weight",
        "features": [
            ("Weighted A+B",            "atomic_weight_weighted_A+B"),
            ("A/B",                   "atomic_weight_A/B"),
            ("A-B",                   "atomic_weight_A-B"),
        ],
    },
    "period_group": {
        "label": "Period & Group",
        "features": [
            ("Period A",                "period_A"),
            ("Period B",                "period_B"),
            ("Group A",                 "group_A"),
            ("Group B",                 "group_B"),
            ("Group A-B",               "group_A-B"),
        ],
    },
    "mendeleev": {
        "label": "Mendeleev Numbers",
        "features": [
            ("A",             "Mendeleev_number_A"),
            ("B",             "Mendeleev_number_B"),
            ("A-B",           "Mendeleev_number_A-B"),
        ],
    },
    "valence": {
        "label": "Valence Electrons",
        "features": [
            ("A",               "valencee_total_A"),
            ("B",               "valencee_total_B"),
            ("A-B",             "valencee_total_A-B"),
            ("A+B",             "valencee_total_A+B"),
            ("Weighted A+B",    "valencee_total_weighted_A+B"),
            ("Norm Weighted A+B","valencee_total_weighted_norm_A+B"),
        ],
    },
    "unpaired": {
        "label": "Unpaired Electrons",
        "features": [
            ("A",              "unpaired_electrons_A"),
            ("B",              "unpaired_electrons_B"),
            ("A-B",            "unpaired_electrons_A-B"),
            ("A+B",            "unpaired_electrons_A+B"),
            ("Weighted A+B",   "unpaired_electrons_weighted_A+B"),
            ("Norm Weighted A+B","unpaired_electrons_weighted_norm_A+B"),
        ],
    },
    "gilman": {
        "label": "Gilman",
        "features": [
            ("A",                "Gilman_A"),
            ("B",                "Gilman_B"),
            ("A-B",              "Gilman_A-B"),
            ("A+B",              "Gilman_A+B"),
            ("Weighted A+B",     "Gilman_weighted_A+B"),
            ("Norm Weighted A+B","Gilman_weighted_norm_A+B"),
        ],
    },
    "z_eff": {
        "label": "Effective Nuclear Charge",
        "features": [
            ("A",                 "Z_eff_A"),
            ("B",                 "Z_eff_B"),
            ("A-B",               "Z_eff_A-B"),
            ("A/B",               "Z_eff_A/B"),
            ("Max",               "Z_eff_max"),
            ("Min",               "Z_eff_min"),
            ("Avg",               "Z_eff_avg"),
            ("Norm Weighted A+B", "Z_eff_weighted_norm_A+B"),
        ],
    },
    "ionization_energy": {
        "label": "Ionization Energy",
        "features": [
            ("A",         "ionization_energy_A"),
            ("B",         "ionization_energy_B"),
            ("A-B",       "ionization_energy_A-B"),
            ("A/B",       "ionization_energy_A/B"),
            ("Max",       "ionization_energy_max"),
            ("Min",       "ionization_energy_min"),
            ("Avg",       "ionization_energy_avg"),
            ("Norm Weighted A+B","ionization_energy_weighted_norm_A+B"),
        ],
    },
    "coordination_number": {
        "label": "Coordination Number",
        "features": [
            ("A",   "coordination_number_A"),
            ("B",   "coordination_number_B"),
            ("A-B", "coordination_number_A-B"),
        ],
    },
    "ratio_closest": {
        "label": "Ratio Closest",
        "features": [
            ("A",         "ratio_closest_A"),
            ("B",         "ratio_closest_B"),
            ("Max",       "ratio_closest_max"),
            ("Min",       "ratio_closest_min"),
            ("Avg",       "ratio_closest_avg"),
        ],
    },
    "polyhedron_distortion": {
        "label": "Polyhedron Distortion",
        "features": [
            ("A",            "polyhedron_distortion_A"),
            ("B",            "polyhedron_distortion_B"),
            ("Max",          "polyhedron_distortion_max"),
            ("Min",          "polyhedron_distortion_min"),
            ("Avg",          "polyhedron_distortion_avg"),
        ],
    },
    "cif_radius": {
        "label": "CIF Radius",
        "features": [
            ("A",            "CIF_radius_A"),
            ("B",            "CIF_radius_B"),
            ("A/B",          "CIF_radius_A/B"),
            ("A-B",          "CIF_radius_A-B"),
            ("Avg",          "CIF_radius_avg"),
            ("Norm Weighted A+B","CIF_radius_weighted_norm_A+B"),
        ],
    },
    "pauling_radius_cn12": {
        "label": "Pauling Radius (CN12)",
        "features": [
            ("A",   "Pauling_radius_CN12_A"),
            ("B",   "Pauling_radius_CN12_B"),
            ("A-B", "Pauling_radius_CN12_A-B"),
            ("A/B", "Pauling_radius_CN12_A/B"),
            ("Avg", "Pauling_radius_CN12_avg"),
            ("Norm Weighted A+B","Pauling_radius_CN12_weighted_norm_A+B"),
        ],
    },
    "pauling_en": {
        "label": "Pauling Electronegativity",
        "features": [
            ("A",            "Pauling_EN_A"),
            ("B",            "Pauling_EN_B"),
            ("A-B",          "Pauling_EN_A-B"),
            ("A/B",          "Pauling_EN_A/B"),
            ("Max",          "Pauling_EN_max"),
            ("Min",          "Pauling_EN_min"),
            ("Avg",          "Pauling_EN_avg"),
            ("Norm Weighted A+B","Pauling_EN_weighted_norm_A+B"),
        ],
    },
    "martynov_batsanov_en": {
        "label": "Martynov–Batsanov EN",
        "features": [
            ("A",       "Martynov_Batsanov_EN_A"),
            ("B",       "Martynov_Batsanov_EN_B"),
            ("A-B",     "Martynov_Batsanov_EN_A-B"),
            ("A/B",     "Martynov_Batsanov_EN_A/B"),
            ("Max",     "Martynov_Batsanov_EN_max"),
            ("Min",     "Martynov_Batsanov_EN_min"),
            ("Avg",     "Martynov_Batsanov_EN_avg"),
            ("Norm Weighted A+B","Martynov_Batsanov_EN_weighted_norm_A+B"),
        ],
    },
    "melting_point": {
        "label": "Melting Point (K)",
        "features": [
            ("A",       "melting_point_K_A"),
            ("B",       "melting_point_K_B"),
            ("A-B",     "melting_point_K_A-B"),
            ("A/B",     "melting_point_K_A/B"),
            ("Max",     "melting_point_K_max"),
            ("Min",     "melting_point_K_min"),
            ("Avg",     "melting_point_K_avg"),
            ("Norm Weighted A+B","melting_point_K_weighted_norm_A+B"),
        ],
    },
    "density": {
        "label": "Density",
        "features": [
            ("A",               "density_A"),
            ("B",               "density_B"),
            ("A-B",             "density_A-B"),
            ("A/B",             "density_A/B"),
            ("Max",             "density_max"),
            ("Min",             "density_min"),
            ("Avg",             "density_avg"),
            ("Norm Weighted A+B","density_weighted_norm_A+B"),
        ],
    },
    "specific_heat": {
        "label": "Specific Heat",
        "features": [
            ("A",         "specific_heat_A"),
            ("B",         "specific_heat_B"),
            ("A-B",       "specific_heat_A-B"),
            ("A/B",       "specific_heat_A/B"),
            ("Max",       "specific_heat_max"),
            ("Min",       "specific_heat_min"),
            ("Avg",       "specific_heat_avg"),
            ("Norm Weighted A+B","specific_heat_weighted_norm_A+B"),
        ],
    },
    "cohesive_energy": {
        "label": "Cohesive Energy",
        "features": [
            ("A",       "cohesive_energy_A"),
            ("B",       "cohesive_energy_B"),
            ("A-B",     "cohesive_energy_A-B"),
            ("A/B",     "cohesive_energy_A/B"),
            ("Max",     "cohesive_energy_max"),
            ("Min",     "cohesive_energy_min"),
            ("Avg",     "cohesive_energy_avg"),
            ("Norm Weighted A+B","cohesive_energy_weighted_norm_A+B"),
        ],
    },
    "bulk_modulus": {
        "label": "Bulk Modulus",
        "features": [
            ("A",          "bulk_modulus_A"),
            ("B",          "bulk_modulus_B"),
            ("A-B",        "bulk_modulus_A-B"),
            ("A/B",        "bulk_modulus_A/B"),
            ("Max",        "bulk_modulus_max"),
            ("Min",        "bulk_modulus_min"),
            ("Avg",        "bulk_modulus_avg"),
            ("Norm Weighted A+B",  "bulk_modulus_weighted_norm_A+B"),
        ],
    },
}


feature_groups_pca = {
    "basic_atomic_properties": {
        "label": "Basic Atomic Properties",
        "features": [
            ("Atomic weight", "Atomic weight"),
            ("Atomic number", "Atomic number"),
            ("Period",         "Period"),
            ("Group",          "Group"),
            ("Families",       "Families"),
        ],
    },
    "electronic_structure": {
        "label": "Electronic Structure",
        "features": [
            ("Mendeleev number",               "Mendeleev number"),
            ("Quantum number l",               "quantum  number l"),
            ("Valence s",                      "valence s"),
            ("Valence p",                      "valence p"),
            ("Valence d",                      "valence d"),
            ("Valence f",                      "valence f"),
            ("Unfilled s",                     "unfilled s"),
            ("Unfilled p",                     "unfilled p"),
            ("Unfilled d",                     "unfilled d"),
            ("Onfilled f",                     "unfilled f"),
            ("Valence electrons",       "no. of  valence  electrons"),
            ("Outer shell electrons",          "outer shell electrons"),
            ("Gilman valence electrons","Gilman no. of valence electrons"),
            ("Metallic valence",               "Metallic  valence"),
            ("Zeff",                           "Zeff"),
            ("1st Bohr radius",           "1st Bohr radius (a0)"),
        ],
    },
    "electronegativity_and_electron_affinity": {
        "label": "Electronegativity & Electron Affinity",
        "features": [
            ("Ionization energy",           "Ionization energy (eV)"),
            ("Electron affinity",           "Electron affinity (ev)"),
            ("Pauling EN",                       "Pauling EN"),
            ("Martynov-Batsanov EN",             "Martynov Batsanov EN"),
            ("Mulliken EN",                      "Mulliken EN"),
            ("Allred EN",                        "Allred EN"),
            ("Allred-Rockow EN",                 "Allred Rockow EN"),
            ("Nagle EN",                         "Nagle EN"),
            ("Ghosh EN",                         "Ghosh EN"),
        ],
    },
    "atomic_and_ionic_radii": {
        "label": "Atomic & Ionic Radii",
        "features": [
            ("Atomic radius calculated",   "Atomic radius calculated"),
            ("Covalent radius",            "Covalent radius"),
            ("Ionic radius",               "Ionic radius"),
            ("Effective ionic radius",     "Effective ionic radius"),
            ("Miracle radius",             "Miracle radius"),
            ("van der Waals radius",       "van der Waals radius"),
            ("Zunger radii sum",           "Zunger radii sum"),
            ("Crystal radius",             "Crystal radius"),
            ("Covalent CSD radius",        "Covalent CSD radius"),
            ("Slater radius",              "Slater radius"),
            ("Orbital radius",             "Orbital radius"),
            ("Polarizability",        "polarizability, A^3"),
        ],
    },
    "thermal_and_physical_properties": {
        "label": "Thermal & Physical Properties",
        "features": [
            ("Melting point",          "Melting point, K"),
            ("Boiling point",          "Boiling point, K"),
            ("Density, g/mL",             "Density,  g/mL"),
            ("Specific heat",      "Specific heat, J/g K"),
            ("Heat of fusion",    "Heat of fusion,  kJ/mol"),
            ("Heat of vaporization",    "Heat of vaporization,  kJ/mol"),
            ("Heat of atomization",     "Heat of atomization,  kJ/mol"),
            ("Thermal conductivity",     "Thermal conductivity, W/m K"),
            ("Cohesive energy",           "Cohesive  energy"),
            ("Bulk modulus",         "Bulk modulus, GPa"),
        ],
    },
    "dft_lsd_and_lda_properties": {
        "label": "DFT LSD & LDA Properties",
        "features": [
            ("DFT LDA Etot",  "DFT LDA Etot"),
            ("DFT LDA Ekin",  "DFT LDA Ekin"),
            ("DFT LDA Ecoul", "DFT LDA Ecoul"),
            ("DFT LDA Eenuc", "DFT LDA Eenuc"),
            ("DFT LDA Exc",   "DFT LDA Exc"),
            ("DFT LSD Etot",  "DFT LSD Etot"),
            ("DFT LSD Ekin",  "DFT LSD Ekin"),
            ("DFT LSD Ecoul", "DFT LSD Ecoul"),
            ("DFT LSD Eenuc", "DFT LSD Eenuc"),
            ("DFT LSD Exc",   "DFT LSD Exc"),
        ],
    },
    "dft_rlda_and_scrlda_properties": {
        "label": "DFT RLDA & ScRLDA Properties",
        "features": [
            ("DFT RLDA Etot",   "DFT RLDA Etot"),
            ("DFT RLDA Ekin",   "DFT RLDA Ekin"),
            ("DFT RLDA Ecoul",  "DFT RLDA Ecoul"),
            ("DFT RLDA Eenuc",  "DFT RLDA Eenuc"),
            ("DFT RLDA Exc",    "DFT RLDA Exc"),
            ("DFT ScRLDA Etot", "DFT ScRLDA Etot"),
            ("DFT ScRLDA Ekin", "DFT ScRLDA Ekin"),
            ("DFT ScRLDA Ecoul","DFT ScRLDA Ecoul"),
            ("DFT ScRLDA Eenuc","DFT ScRLDA Eenuc"),
            ("DFT ScRLDA Exc",  "DFT ScRLDA Exc"),
        ],
    },
}

feature_groups_cluster = {
    "basic_atomic_properties": {
        "label": "Basic Atomic Properties",
        "features": [
            ("Atomic weight", "Atomic weight"),
            ("Atomic number", "Atomic number"),
            ("Period",         "Period"),
            ("Group",          "Group"),
            ("Families",       "Families"),
        ],
    },
    "electronic_structure": {
        "label": "Electronic Structure",
        "features": [
            ("Mendeleev number",               "Mendeleev number"),
            ("Quantum number l",               "quantum  number l"),
            ("Valence s",                      "valence s"),
            ("Valence p",                      "valence p"),
            ("Valence d",                      "valence d"),
            ("Valence f",                      "valence f"),
            ("Unfilled s",                     "unfilled s"),
            ("Unfilled p",                     "unfilled p"),
            ("Unfilled d",                     "unfilled d"),
            ("Onfilled f",                     "unfilled f"),
            ("Valence electrons",       "no. of  valence  electrons"),
            ("Outer shell electrons",          "outer shell electrons"),
            ("Gilman valence electrons","Gilman no. of valence electrons"),
            ("Metallic valence",               "Metallic  valence"),
            ("Zeff",                           "Zeff"),
            ("1st Bohr radius",           "1st Bohr radius (a0)"),
        ],
    },
    "electronegativity_and_electron_affinity": {
        "label": "Electronegativity & Electron Affinity",
        "features": [
            ("Ionization energy",           "Ionization energy (eV)"),
            ("Electron affinity",           "Electron affinity (ev)"),
            ("Pauling EN",                       "Pauling EN"),
            ("Martynov-Batsanov EN",             "Martynov Batsanov EN"),
            ("Mulliken EN",                      "Mulliken EN"),
            ("Allred EN",                        "Allred EN"),
            ("Allred-Rockow EN",                 "Allred Rockow EN"),
            ("Nagle EN",                         "Nagle EN"),
            ("Ghosh EN",                         "Ghosh EN"),
        ],
    },
    "atomic_and_ionic_radii": {
        "label": "Atomic & Ionic Radii",
        "features": [
            ("Atomic radius calculated",   "Atomic radius calculated"),
            ("Covalent radius",            "Covalent radius"),
            ("Ionic radius",               "Ionic radius"),
            ("Effective ionic radius",     "Effective ionic radius"),
            ("Miracle radius",             "Miracle radius"),
            ("van der Waals radius",       "van der Waals radius"),
            ("Zunger radii sum",           "Zunger radii sum"),
            ("Crystal radius",             "Crystal radius"),
            ("Covalent CSD radius",        "Covalent CSD radius"),
            ("Slater radius",              "Slater radius"),
            ("Orbital radius",             "Orbital radius"),
            ("Polarizability",        "polarizability, A^3"),
        ],
    },
    "thermal_and_physical_properties": {
        "label": "Thermal & Physical Properties",
        "features": [
            ("Melting point",          "Melting point, K"),
            ("Boiling point",          "Boiling point, K"),
            ("Density, g/mL",             "Density,  g/mL"),
            ("Specific heat",      "Specific heat, J/g K"),
            ("Heat of fusion",    "Heat of fusion,  kJ/mol"),
            ("Heat of vaporization",    "Heat of vaporization,  kJ/mol"),
            ("Heat of atomization",     "Heat of atomization,  kJ/mol"),
            ("Thermal conductivity",     "Thermal conductivity, W/m K"),
            ("Cohesive energy",           "Cohesive  energy"),
            ("Bulk modulus",         "Bulk modulus, GPa"),
        ],
    },
    "dft_lsd_and_lda_properties": {
        "label": "DFT LSD & LDA Properties",
        "features": [
            ("DFT LDA Etot",  "DFT LDA Etot"),
            ("DFT LDA Ekin",  "DFT LDA Ekin"),
            ("DFT LDA Ecoul", "DFT LDA Ecoul"),
            ("DFT LDA Eenuc", "DFT LDA Eenuc"),
            ("DFT LDA Exc",   "DFT LDA Exc"),
            ("DFT LSD Etot",  "DFT LSD Etot"),
            ("DFT LSD Ekin",  "DFT LSD Ekin"),
            ("DFT LSD Ecoul", "DFT LSD Ecoul"),
            ("DFT LSD Eenuc", "DFT LSD Eenuc"),
            ("DFT LSD Exc",   "DFT LSD Exc"),
        ],
    },
    "dft_rlda_and_scrlda_properties": {
        "label": "DFT RLDA & ScRLDA Properties",
        "features": [
            ("DFT RLDA Etot",   "DFT RLDA Etot"),
            ("DFT RLDA Ekin",   "DFT RLDA Ekin"),
            ("DFT RLDA Ecoul",  "DFT RLDA Ecoul"),
            ("DFT RLDA Eenuc",  "DFT RLDA Eenuc"),
            ("DFT RLDA Exc",    "DFT RLDA Exc"),
            ("DFT ScRLDA Etot", "DFT ScRLDA Etot"),
            ("DFT ScRLDA Ekin", "DFT ScRLDA Ekin"),
            ("DFT ScRLDA Ecoul","DFT ScRLDA Ecoul"),
            ("DFT ScRLDA Eenuc","DFT ScRLDA Eenuc"),
            ("DFT ScRLDA Exc",  "DFT ScRLDA Exc"),
        ],
    },
}

# ——————————————
# Build controls
# ——————————————
# Action buttons
controls_row_plsda = ui.div(
    ui.input_action_button("select_all",   "Select All"),
    ui.input_action_button("deselect_all", "Deselect All"),
    style="display:flex; gap:8px; margin-bottom:12px;"
)
feature_cards = []
for gid, info in feature_groups_plsda.items():
    # build the per‐feature checkboxes
    checks = [
        ui.input_checkbox(
            f"feat_{make_safe_id(col)}",
            disp,
            value=True,
        )
        for disp, col in info["features"]
    ]
    # build a header that has both the group‐level checkbox and the bold label
    header = ui.card_header(
        ui.div(
            # tell the checkbox itself to only be as wide as it needs to be:
            ui.input_checkbox(
                f"group_{gid}",
                "",              # no label text here
                value=True,
                width="2em"      # make the input only 1em wide
            ),
            # the feature‐group name right after it, no margin or padding
            ui.HTML(f"<b style='margin:0; padding:0;'>{info['label']}</b>"),
            # container flexbox: zero gap so they sit flush
            style="display:flex; align-items:center; gap:0; margin:0; padding:0;"
        ),
        style="font-size:1em;"
    )
    feature_cards.append(
        ui.card(
            header,
            *checks,
            full_screen=False,
        )
    )

# now split into 4 roughly‐equal columns of cards
n_cards = len(feature_cards)
chunk = -(-n_cards // 4)
cards_pls_col1 = feature_cards[0:chunk]
cards_pls_col2 = feature_cards[chunk:2*chunk]
cards_pls_col3 = feature_cards[2*chunk:3*chunk]
cards_pls_col4 = feature_cards[3*chunk:]

i = 1   # for example, the second card in column 3
moved = cards_pls_col3.pop(i)
cards_pls_col4.append(moved)

feature_cards_pca = []
for gid, info in feature_groups_pca.items():
    # Group‐level checkbox + label in the header
    header = ui.card_header(
        ui.div(
            ui.input_checkbox(
                f"pca_group_{gid}",  # group toggle ID
                "",                  # no visible label
                value=True,
                width="2em",
            ),
            ui.HTML(f"<b style='margin:0; padding:0;'>{info['label']}</b>"),
            style="display:flex; align-items:center; gap:0; margin:0; padding:0;"
        ),
        style="font-size:1em;"
    )

    # One checkbox per feature in this group
    checks = [
        ui.input_checkbox(
            f"pca_feat_{make_safe_id(col)}",  # feature‐ID
            disp,                            # display name
            value=True
        )
        for disp, col in info["features"]
    ]

    feature_cards_pca.append(
        ui.card(
            header,
            *checks,
            full_screen=False
        )
    )

# Now split into four roughly‐equal columns
n = len(feature_cards_pca)
chunk = -(-n // 4)
cards_pca_col1 = feature_cards_pca[        :chunk]
cards_pca_col2 = feature_cards_pca[   chunk :2*chunk]
cards_pca_col3 = feature_cards_pca[2*chunk :3*chunk]
cards_pca_col4 = feature_cards_pca[3*chunk :      ]

feature_cards_cluster = []
for gid, info in feature_groups_cluster.items():
    header = ui.card_header(
        ui.div(
            ui.input_checkbox(
                f"clust_group_{gid}",  # <-- new prefix here
                "",
                value=True,
                width="2em",
            ),
            ui.HTML(f"<b style='margin:0; padding:0;'>{info['label']}</b>"),
            style="display:flex; align-items:center; gap:0; margin:0; padding:0;"
        ),
        style="font-size:1em;"
    )
    checks = [
        ui.input_checkbox(
            f"clust_feat_{make_safe_id(col)}",  # <-- and here
            disp,
            value=True
        )
        for disp, col in info["features"]
    ]
    feature_cards_cluster.append(
        ui.card(header, *checks, full_screen=False)
    )

# split into four columns exactly as you did before
n = len(feature_cards_cluster)
chunk = -(-n // 4)
cards_cluster_col1 = feature_cards_cluster[         :chunk]
cards_cluster_col2 = feature_cards_cluster[    chunk:2*chunk]
cards_cluster_col3 = feature_cards_cluster[2*chunk:3*chunk]
cards_cluster_col4 = feature_cards_cluster[3*chunk:       ]

# ——————————————
# UI definition
# ——————————————
app_ui = ui.page_fluid(
    ui.navset_card_tab(
        ui.nav_panel(
            "PCA",
            ui.page_navbar(
                ui.nav_panel(
                "Element Mapping",
                ui.row(
                    # LEFT half: file upload + select/deselect + feature-cards
                    ui.column(
                    6,
                    # select/deselect buttons
                    ui.div(
                        ui.input_action_button("pca_select_all",   "Select All"),
                        ui.input_action_button("pca_deselect_all", "Deselect All"),
                        style="display:flex; gap:8px; margin:12px 0;"
                    ),
                    # now the cards, 4 columns
                    ui.row(
                        ui.column(3, *cards_pca_col1),
                        ui.column(3, *cards_pca_col2),
                        ui.column(3, *cards_pca_col3),
                        ui.column(3, *cards_pca_col4),
                    )
                    ),
                    # RIGHT half: run button + scatter + contributions table
                    ui.column(
                    6,
                    ui.div(
                        output_widget("pca_plot"),
                        style="display:flex; justify-content:center; margin-top:12px;"
                    ),
                    ui.div(
                        ui.output_data_frame("pca_contrib_table"),
                        style="display:flex; justify-content:center; margin-top:12px;"
                        ),
                    )
                )
            ),
                ui.nav_panel(
                "Clustering",
                ui.row(
                    # LEFT: file upload + select/deselect + same feature cards
                    ui.column(
                    6,
                    # buttons to select/deselect the same PCA‐features
                    ui.div(
                        ui.input_action_button("clust_select_all",   "Select All"),
                        ui.input_action_button("clust_deselect_all", "Deselect All"),
                        style="display:flex; gap:8px; margin:12px 0;"
                    ),
                    # your PCA‐feature cards (reuse cards_pca_col1…4)
                    ui.row(
                        ui.column(3, *cards_cluster_col1),
                        ui.column(3, *cards_cluster_col2),
                        ui.column(3, *cards_cluster_col3),
                        ui.column(3, *cards_cluster_col4),
                    )
                    ),
                    # RIGHT: the PCA scatter + contributions
                    ui.column(
                    6,
                    ui.div(output_widget("clust_plot"),
                            style="display:flex; justify-content:center; margin-top:12px;"),
                    ui.div(ui.output_data_frame("clust_contrib_table"),
                            style="display:flex; justify-content:center; margin-top:12px;")
                        )
                    )
                )
            )
        ),
        ui.nav_panel(
            "PLS-DA",
            ui.page_navbar(
                ui.nav_panel(
                "Visualization",
                ui.row(
                    # LEFT HALF: each card in its own bordered box
                        ui.column(
                        6,
                        controls_row_plsda,
                        ui.row(
                            ui.column(3, *cards_pls_col1),
                            ui.column(3, *cards_pls_col2),
                            ui.column(3, *cards_pls_col3),
                            ui.column(3, *cards_pls_col4),
                        )
                        ),
                    # RIGHT HALF: unchanged
                        ui.column(
                        6,
                        ui.div(output_widget("pls_plot"), style="display:flex; justify-content:center;"),
                        ui.row(
                            ui.column(4, ui.output_data_frame("metrics_table")),
                            ui.column(7, ui.output_data_frame("contrib_table")),
                            style="margin-top: 5px;"
                            )
                        )
                    )
                ),
                ui.nav_panel(
                    "Evaluation",

                    # 1) CV vs # Components, centered
                    ui.h3("Number of Components"),
                    ui.input_action_button("run_eval_n", "Run"),
                    ui.row(
                        ui.column(
                        8,
                        output_widget("eval_n_plot"),
                        offset=2,
                        style="display:flex; justify-content:center;"
                        )
                    ),
                    ui.hr(),

                   # 2) Forward Feature Selection
                    ui.h3("Forward Feature Selection"),
                    ui.input_action_button("run_forward", "Run"),
                    ui.row(
                        # Left half: perf plot + text stacked
                        ui.column(
                            6,
                            ui.div(
                                ui.div(output_widget("forward_perf_plot"), style="flex:1;"),
                                ui.div(ui.output_text_verbatim("forward_log"),
                                    style="flex:1; overflow:auto;"),
                                style="display:flex; flex-direction:column; height:500px;"
                            )
                        ),
                        # Right half: scatter + slider stacked to exactly 500px
                        ui.column(
                        6,
                        ui.div(
                            # make the whole column a centered column
                            ui.div(
                            output_widget("forward_scatter_plot"),
                            style="flex:1; display:flex; justify-content:center; align-items:center;"
                            ),
                            ui.div(
                            ui.output_ui("forward_slider_ui"),
                            style="flex:none; display:flex; justify-content:center; margin-top:8px;"
                            ),
                            style="display:flex; flex-direction:column; align-items:center; height:600px;"
                        )
                        )
                    ),
                    ui.hr(),

                    # 3) Backward Feature Selection
                    ui.h3("Backward Feature Selection"),
                    ui.input_action_button("run_backward", "Run"),
                    ui.row(
                        # Left half: perf plot + text stacked
                        ui.column(
                            6,
                            ui.div(
                                ui.div(output_widget("backward_perf_plot"), style="flex:1;"),
                                ui.div(ui.output_text_verbatim("backward_log"),
                                    style="flex:1; overflow:auto;"),
                                style="display:flex; flex-direction:column; height:500px;"
                            )
                        ),
                        # Right half: scatter + slider stacked to exactly 500px
                        ui.column(
                        6,
                        ui.div(
                            ui.div(
                            output_widget("backward_scatter_plot"),
                            style="flex:1; display:flex; justify-content:center; align-items:center;"
                            ),
                            ui.div(
                            ui.output_ui("backward_slider_ui"),
                            style="flex:none; display:flex; justify-content:center; margin-top:8px;"
                            ),
                            style="display:flex; flex-direction:column; align-items:center; height:600px;"
                        )
                        )
                    ),
                )
            )
        )
    )
)

# ——————————————
# Server logic
# ——————————————
def server(input, output, session):
    
    # ——————————————

    # PCA calculation

    # ——————————————
    # Select/Deselect all groups
    @reactive.Effect
    @reactive.event(input.pca_select_all)
    def select_all():
        for gid in feature_groups_pca:
            ui.update_checkbox(f"pca_group_{gid}", value=True, session=session)

    @reactive.Effect
    @reactive.event(input.pca_deselect_all)
    def deselect_all():
        for gid in feature_groups_pca:
            ui.update_checkbox(f"pca_group_{gid}", value=False, session=session)

    # When a group toggles, toggle its features
    for gid, info in feature_groups_pca.items():
        @reactive.Effect
        @reactive.event(getattr(input, f"pca_group_{gid}"))
        def _grp_toggle(gid=gid, info=info):
            state = getattr(input, f"pca_group_{gid}")()
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"pca_feat_{fid}", value=state, session=session)

    @reactive.Calc
    def pca_res():
        data = pd.read_excel(pca_data)

        # 1) pull out only the user‐selected columns
        sel = [col
            for _, info in feature_groups_pca.items()
            for disp, col in info["features"]
            if getattr(input, f"pca_feat_{make_safe_id(col)}")()]
        sel = [c for c in sel if c in data.columns]
        if not sel:
            return None

        # 2) build your DataFrame, drop NaN‐columns *first*, then drop zero‐variance
        Xdf = data[sel].copy()
        Xdf = Xdf.dropna(axis=1, how="any")              # ← drop any column with at least one NaN
        zero_var = Xdf.std(axis=0) == 0
        if zero_var.any():
            Xdf = Xdf.loc[:, ~zero_var]
        if Xdf.shape[1] < 2:
            return None   # not enough valid features

        # 3) scale and PCA as before
        X = StandardScaler().fit_transform(Xdf)
        pca = PCA(n_components=2, random_state=42).fit(X)
        pcs = pca.transform(X)

        # 4) Build a DataFrame for plotting
        dfp = pd.DataFrame({
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "Symbol": data["Symbol"].astype(str)
        })

        # 5) Explained‐variance
        pc1_pct = pca.explained_variance_ratio_[0] * 100
        pc2_pct = pca.explained_variance_ratio_[1] * 100

        # 6) Top‐10 loadings
        comps = pca.components_
        def top10(component):
            loadings = list(zip(sel, component))
            loadings.sort(key=lambda x: abs(x[1]), reverse=True)
            return loadings[:10]

        top1 = top10(comps[0])
        top2 = top10(comps[1])

        contrib_df = pd.DataFrame({
            "PC1 Feature": [f for f, _ in top1],
            "PC1 Loading": [f"{v:.3f}" for _, v in top1],
            "PC2 Feature": [f for f, _ in top2],
            "PC2 Loading": [f"{v:.3f}" for _, v in top2],
        })

        return {
            "dfp": dfp,
            "pc1_pct": pc1_pct,
            "pc2_pct": pc2_pct,
            "contrib": contrib_df
        }

    @render_widget
    def pca_plot():
        res = pca_res()
        if res is None:
            fig = go.Figure()
            fig.update_layout(
                title="No features selected or no data.",
                template="ggplot2", width=800, height=800
            )
            return fig

        # 1) assign each point to a group (or "other")
        dfp = res["dfp"].copy()
        dfp["Group"] = dfp["Symbol"].map(symbol_to_group).fillna("other")

        # 2) scatter with color by Group
        fig = px.scatter(
            dfp,
            x="PC1", y="PC2", text="Symbol",
            color="Group",
            color_discrete_map={
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
            },
            labels={
              "PC1": f"PC 1 ({res['pc1_pct']:.1f}%)",
              "PC2": f"PC 2 ({res['pc2_pct']:.1f}%)"
            },
            template="ggplot2"
        )
        fig.update_traces(marker=dict(size=26, opacity=0.6))
        fig.update_layout(
            width=800, height=800,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=False,
            font=dict(size=16)
        )
        return fig
    @render.data_frame
    def pca_contrib_table():
        res = pca_res()
        if res is None:
            return pd.DataFrame({"Message": ["No PCA results."]})
        return res["contrib"]    
    

    # ——————————————

    # PCA clustering calculation

    # ——————————————

   # 1) “Select all” / “Deselect all” for clustering
    @reactive.Effect
    @reactive.event(input.clust_select_all)
    def _():
        for gid in feature_groups_cluster:
            ui.update_checkbox(f"clust_group_{gid}", value=True, session=session)

    @reactive.Effect
    @reactive.event(input.clust_deselect_all)
    def _():
        for gid in feature_groups_cluster:
            ui.update_checkbox(f"clust_group_{gid}", value=False, session=session)

    # 2) When any clust_group_* toggles, mirror into its clust_feat_* children
    for gid, info in feature_groups_cluster.items():
        @reactive.Effect
        @reactive.event(getattr(input, f"clust_group_{gid}"))
        def _grp_toggle(gid=gid, info=info):
            state = getattr(input, f"clust_group_{gid}")()
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"clust_feat_{fid}", value=state, session=session)
    
    
    @reactive.Calc
    def clust_res():
        data = pd.read_excel(pca_data)

        sel = [col for _, info in feature_groups_pca.items()
                for disp, col in info["features"]
                if getattr(input, f"clust_feat_{make_safe_id(col)}")()]
        sel = [c for c in sel if c in data.columns]
        if not sel:
            return None

        # **DROP zero-variance columns** from your PCA matrix
        Xdf = data[sel].copy()
        zero_var = Xdf.std(axis=0) == 0
        if zero_var.any():
            Xdf = Xdf.loc[:, ~zero_var]
        if Xdf.shape[1] < 2:
            return None  # not enough features for 2 components

        # then proceed
        X = StandardScaler().fit_transform(Xdf)
        pca = PCA(n_components=2, random_state=42).fit(X)
        pcs = pca.transform(X)
        dfp = pd.DataFrame({
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "Symbol": data["Symbol"].astype(str)
        })
        pc1_pct = pca.explained_variance_ratio_[0] * 100
        pc2_pct = pca.explained_variance_ratio_[1] * 100

        # build the top-10 loading table just like in pca_res()
        comps = pca.components_
        def top10(component):
            loadings = list(zip(sel, component))
            loadings.sort(key=lambda x: abs(x[1]), reverse=True)
            return loadings[:10]

        top1 = top10(comps[0])
        top2 = top10(comps[1])
        contrib_df = pd.DataFrame({
            "PC1 Feature": [f for f, _ in top1],
            "PC1 Loading": [f"{v:.3f}" for _, v in top1],
            "PC2 Feature": [f for f, _ in top2],
            "PC2 Loading": [f"{v:.3f}" for _, v in top2],
        })

        # b) read hard-coded structure file
        struct_df = pd.read_excel(cluster_data, usecols=["Formula", "Structure type"])
        import re
        def split_formula(f):
            return re.findall(r"[A-Z][a-z]?", f)

        links = []
        for _, row in struct_df.iterrows():
            a, b = split_formula(row["Formula"])
            try:
                x0, y0 = dfp.loc[dfp.Symbol==a, ["PC1","PC2"]].iloc[0]
                x1, y1 = dfp.loc[dfp.Symbol==b, ["PC1","PC2"]].iloc[0]
            except IndexError:
                continue
            links.append({
                "x": [x0, x1],
                "y": [y0, y1],
                "mid_x": (x0 + x1) / 2,
                "mid_y": (y0 + y1) / 2,
                "struct": row["Structure type"]
            })

        return {
            "dfp": dfp,
            "pc1_pct": pc1_pct,
            "pc2_pct": pc2_pct,
            "contribution": contrib_df,
            "links": links
        }
    # 4) render the clustering plot
    @render_widget
    def clust_plot():
        res = clust_res()
        if not res:
            fig = go.Figure()
            fig.update_layout(title="No data", width=800, height=800)
            return fig

        dfp = res["dfp"]
        fig = px.scatter(dfp, x="PC1", y="PC2", text="Symbol", template="ggplot2")
        fig.update_traces(marker=dict(color="green", size=12, opacity=0.6))

        # colour map for your three structure types
        cmap = {
            "CsCl": "#c3121e",
            "NaCl": "#0348a1",
            "ZnS":  "#ffb01c",
        }
        fig.update_traces(marker=dict(size=26, opacity=0.6))
        seen = set()
        for link in res["links"]:
            struct = link["struct"]
            col = cmap.get(struct, "#888888")

            # 1) always draw the line (no legend)
            fig.add_trace(go.Scatter(
                x=link["x"],
                y=link["y"],
                mode="lines",
                line=dict(color=col, width=1),
                showlegend=False
            ))
            
            # 2) draw the midpoint, but only give a legend entry the first time
            show_leg = struct not in seen
            fig.add_trace(go.Scatter(
                x=[link["mid_x"]],
                y=[link["mid_y"]],
                mode="markers",
                marker=dict(color=col, size=10),
                name=struct,
                legendgroup=struct,
                showlegend=show_leg
            ))
            seen.add(struct)
            fig.update_traces(marker=dict(opacity=0.6))

        
        fig.update_layout(
            width=800, height=700,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            font=dict(size=16),
            showlegend=True
        )
        return fig
    @render.data_frame
    def clust_contrib_table():
        res = clust_res()
        if res is None:
            return pd.DataFrame({"Message": ["No PCA results."]})
        return res["contribution"] 
    # ——————————————

    # PLS-DA calculation

    # ——————————————
    
    # Select/Deselect all groups
    @reactive.Effect
    @reactive.event(input.select_all)
    def _select_all():
        for gid in feature_groups_plsda:
            ui.update_checkbox(f"group_{gid}", value=True, session=session)

    @reactive.Effect
    @reactive.event(input.deselect_all)
    def _deselect_all():
        for gid in feature_groups_plsda:
            ui.update_checkbox(f"group_{gid}", value=False, session=session)

    # When a group toggles, toggle its features
    for gid, info in feature_groups_plsda.items():
        @reactive.Effect
        @reactive.event(getattr(input, f"group_{gid}"))
        def _grp_toggle(gid=gid, info=info):
            state = getattr(input, f"group_{gid}")()
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"feat_{fid}", value=state, session=session)

    # Core PLS-DA reactive calculation
    @reactive.Calc
    def run_plsda():
        # 1) Gather selected columns
        sel = []
        for info in feature_groups_plsda.values():
            for _, col in info["features"]:
                if getattr(input, f"feat_{make_safe_id(col)}")():
                    sel.append(col)
        valid = [c for c in sel if c in df.columns]
        if not valid:
            return None

        # 2) Prepare X, y
        X = df[valid].values
        y = df[label_col].values

        # 3) Standardize the full data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 4) Fit PLS‐DA on full data
        #    One‐hot encode y for PLSRegression
        Y_dummy = pd.get_dummies(y)
        pls = PLSRegression(n_components=2, scale=False)
        pls.fit(X_scaled, Y_dummy)

        # 5) Extract scores (x_scores_) for every sample
        scores = pls.x_scores_

        # 6) Explained variance (approximate)
        total_variance = np.var(X_scaled, axis=0).sum()
        var_comp1 = np.var(scores[:, 0])
        var_comp2 = np.var(scores[:, 1])
        pls1_pct = var_comp1 / total_variance * 100
        pls2_pct = var_comp2 / total_variance * 100

        # 7) Classify in the latent space on full data
        #    (use the PLS predictions and take the argmax)
        y_pred_cont = pls.predict(X_scaled)
        pred_idx = np.argmax(y_pred_cont, axis=1)
        pred_labels = [Y_dummy.columns[i] for i in pred_idx]

        # 8) Compute metrics on the full dataset
        acc = accuracy_score(y, pred_labels)
        f1  = f1_score(y, pred_labels, average="macro")
        sil = silhouette_score(scores, y) if len(np.unique(y)) > 1 else np.nan

        # 9) Fisher Discriminant Ratio
        overall_mean = scores.mean(axis=0)
        between_var = 0.0
        within_var = 0.0
        for cls in np.unique(y):
            cls_scores = scores[y == cls]
            m_cls = cls_scores.mean(axis=0)
            between_var += cls_scores.shape[0] * np.sum((m_cls - overall_mean) ** 2)
            within_var  += np.sum((cls_scores - m_cls) ** 2)
        fisher = between_var / within_var if within_var > 1e-6 else np.nan

        # 10) Pack metrics into a DataFrame
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "F1 Score", "Silhouette", "Fisher Ratio"],
            "Value":  [f"{acc:.3f}",
                       f"{f1:.3f}",
                       f"{sil:.3f}" if not np.isnan(sil) else "N/A",
                       f"{fisher:.3f}" if not np.isnan(fisher) else "N/A"]
        })

        # 11) Top‐5 feature contributions for each component
        W = pls.x_weights_
        w1, w2 = W[:, 0], (W[:, 1] if W.shape[1] > 1 else None)
        top_n = min(10, len(valid))
        idx1 = np.argsort(np.abs(w1))[::-1][:top_n]
        top1 = [(valid[i], w1[i]) for i in idx1]

        if w2 is not None:
            idx2 = np.argsort(np.abs(w2))[::-1][:top_n]
            top2 = [(valid[i], w2[i]) for i in idx2]
        else:
            top2 = []

        contrib_rows = []
        for i in range(max(len(top1), len(top2))):
            f1n, v1 = top1[i] if i < len(top1) else ("", "")
            f2n, v2 = top2[i] if i < len(top2) else ("", "")
            contrib_rows.append([
                f1n, f"{v1:.3f}" if v1 != "" else "",
                f2n, f"{v2:.3f}" if v2 != "" else ""
            ])
        contrib_df = pd.DataFrame(
            contrib_rows,
            columns=["PLS1 Feature", "PLS1 Score", "PLS2 Feature", "PLS2 Score"]
        )

        # 12) Return everything for plotting & tables
        return {
            "scores": scores,
            "labels": y,
            "metrics": metrics_df,
            "contrib": contrib_df,
            "pls1_pct": pls1_pct,
            "pls2_pct": pls2_pct
        }

    # ——————————————
    # Plot renderer
    # ——————————————
    @render_widget
    def pls_plot():
        res = run_plsda()
        fig = go.Figure()
        if res is None:
            # no features selected
            fig.update_layout(
                title="PLS-DA Projection (No features selected)",
                template="ggplot2",
                width=800, height=700, autosize=False
            )
            return fig

        df_sc = pd.DataFrame({
            "LV1": res["scores"][:,0],
            "LV2": res["scores"][:,1],
            "Class": res["labels"]
        })
        cmap = {
            cl: c for cl,c in zip(
                sorted(df_sc["Class"].unique()),
                ["#c3121e","#0348a1","#ffb01c","#027608",
                 "#1dace6","#9c5300","#9966cc","#ff4500"]
            )
        }

        fig = px.scatter(
            df_sc,
            x="LV1", y="LV2", color="Class",
            labels={
                "LV1": f"LV 1 ({res['pls1_pct']:.1f}%)",
                "LV2": f"LV 2 ({res['pls2_pct']:.1f}%)"
            },
            template="ggplot2",
            color_discrete_map=cmap
        )
        fig.update_traces(marker=dict(size=26, opacity=0.8))
        fig.update_layout(
            width=800, height=700, autosize=False,
            font=dict(size=18), showlegend=True
        )
        return fig

    # ——————————————
    # Metrics table
    # ——————————————
    @render.data_frame
    def metrics_table():
        res = run_plsda()
        if res is None:
            return pd.DataFrame({"Message":["No features selected."]})
        return res["metrics"]

    # ——————————————
    # Contributions table
    # ——————————————
    @render.data_frame
    def contrib_table():
        res = run_plsda()
        if res is None:
            return pd.DataFrame()
        return res["contrib"]
    





    # ————————————————————————

    # Evaluation section

    # ————————————————————————




    
    # ——————————————————————————————————————
    # 1) Accuracy vs # Components
    # ——————————————————————————————————————
    @reactive.Calc
    @reactive.event(input.run_eval_n)
    def eval_n_res():
        # only runs when the button is clicked
        numeric = df.drop(columns=[label_col])
        return [
            evaluate_subset(numeric, df[label_col], list(numeric.columns), n_components=n)
            for n in range(1, 16)
        ]

    @render_widget
    def eval_n_plot():
        if input.run_eval_n() < 1:
            fig = go.Figure()
            fig.update_layout(
                title="Click “Run” to calculate best number of Components.",
                template="ggplot2",
                width=800, height=300
            )
            return fig

        hist = eval_n_res()
        fig = go.Figure(go.Scatter(
            x=list(range(1, len(hist) + 1)),
            y=hist,
            mode="lines+markers",
            name="Accuracy"
        ))
        fig.update_layout(
            title="PLS-DA: Accuracy vs # Components",
            xaxis_title="Components",
            yaxis_title="Accuracy",
            template="ggplot2",
            width=1200, height=300
        )
        return fig


    # ——————————————————————————————————————
    # 2) Forward Feature Selection
    # ——————————————————————————————————————
    @reactive.Calc
    @reactive.event(input.run_forward)
    def forward_res():
        numeric = df.drop(columns=[label_col])
        target  = df[label_col]
        return forward_selection_plsda_df(
            numeric, target,
            max_features=40, n_components=2, scoring='accuracy'
        )


    @render_widget
    def forward_perf_plot():
        if input.run_forward() < 1:
            fig = go.Figure()
            fig.update_layout(
                title="Waiting to run forward selection…",
                template="ggplot2", width=800, height=300
            )
            return fig

        hist, _ = forward_res()
        # x from 1 to len(hist)
        fig = go.Figure(go.Scatter(
            x=list(range(2, len(hist) + 1)),
            y=hist,
            mode="lines+markers"
        ))
        fig.update_layout(
            title="Forward Selection Accuracy",
            xaxis_title="Number of Features",
            yaxis_title="Accuracy",
            xaxis=dict(tickmode="linear", tick0=2, dtick=1),
            template="ggplot2", width=800, height=300,
            xaxis_range=[1, len(hist)+1] 
        )
        return fig


    @render.text
    def forward_log():
        if input.run_forward() < 1:
            return "Forward selection not run yet."
        _, iters = forward_res()
        return "\n".join(
            f"Step {it} | Features : {it+1} | sel={sel}, score={sc:.4f}"
            for it, sel, sc in iters
        )


    @render.ui
    def forward_slider_ui():
        # before we run, show a harmless 1–1 slider
        if input.run_forward() < 1:
            return ui.input_slider(
                "forward_step", "Step",
                min=1, max=1, value=1, step=1
            )
        hist, _ = forward_res()
        n = len(hist)
        return ui.input_slider(
            "forward_step", "Step",
            min=1, max=n, value=1, step=1
        )


    @render_widget
    def forward_scatter_plot():
        if input.run_forward() < 1:
            fig = go.Figure()
            fig.update_layout(
                title="…waiting for forward selection…",
                width=600, height=500
            )
            return fig

        _, iters = forward_res()
        # convert 1‐based slider back to 0‐based index
        idx = input.forward_step() - 1
        it, sel_feats, sc = iters[idx]

        X = StandardScaler().fit_transform(df[sel_feats])
        Y = pd.get_dummies(df[label_col])
        pls = PLSRegression(n_components=2, scale=False).fit(X, Y)
        scores = pls.x_scores_
        df_sc = pd.DataFrame(scores, columns=["Component1", "Component2"])
        df_sc["Class"] = df[label_col].values

        cmap = {
            cl: c for cl,c in zip(
                sorted(df_sc["Class"].unique()),
                ["#c3121e","#0348a1","#ffb01c","#027608",
                 "#1dace6","#9c5300","#9966cc","#ff4500"]
            )
        }

        fig = px.scatter(
            df_sc, x="Component1", y="Component2", color="Class",
            template="ggplot2", title=f"{it+1} Features", color_discrete_map=cmap
        )
        fig.update_traces(marker=dict(size=26, opacity=0.6))
        fig.update_layout(width=600, height=500)
        return fig


    # ——————————————————————————————————————
    # 3) Backward Elimination
    # ——————————————————————————————————————
    @reactive.Calc
    @reactive.event(input.run_backward)
    def backward_res():
        numeric = df.drop(columns=[label_col])
        target  = df[label_col]
        return backward_elimination_plsda_df(
            numeric, target,
            min_features=120, 
            n_components=2, 
            scoring='accuracy'
        )


    @render_widget
    def backward_perf_plot():
        if input.run_backward() < 1:
            fig = go.Figure()
            fig.update_layout(
                title="Waiting to run backward selection…",
                template="ggplot2", width=800, height=300
            )
            return fig

        hist, iters = backward_res()
        counts = [n for n,_,_ in iters]

        fig = go.Figure(go.Scatter(
            x=counts,
            y=hist,
            mode="lines+markers"
        ))
        fig.update_layout(
            title="Backward Elimination Accuracy",
            xaxis_title="Number of Features",
            yaxis_title="Accuracy",
            template="ggplot2", width=800, height=300,
            # ← replace your old xaxis=… here with:
            xaxis=dict(
                autorange='reversed',      # ← flip the direction
                tickmode='linear',
                tick0=min(counts),
                dtick=1
            )
        )
        return fig

    @render.text
    def backward_log():
        if input.run_backward() < 1:
            return "Backward selection not run yet."
        _, iters = backward_res()

        return "\n".join(
            f"Step {i+1} | Features : {nfeat} | sel={feats}, score={sc:.4f}"
            for i, (nfeat, feats, sc) in enumerate(iters)
        )


    @render.ui
    def backward_slider_ui():
        _, iters = backward_res()
        counts   = [n for n, _, _ in iters]
        high, low = max(counts), min(counts)

        return ui.input_slider(
            "backward_step", "Features",
            # left edge = high count, right = low count
            min=low,
            max=high,
            value=high,
            step=-1
        )

    @render_widget
    def backward_scatter_plot():
        if input.run_backward() < 1:
            fig = go.Figure()
            fig.update_layout(title="…waiting for backward selection…", width=600, height=500)
            return fig

        _, iters = backward_res()
        counts = [n for n,_,_ in iters]
        sel = input.backward_step()
        idx = counts.index(sel)

        nfeat, feats, sc = iters[idx]

        X = StandardScaler().fit_transform(df[feats])
        Y = pd.get_dummies(df[label_col])
        pls = PLSRegression(n_components=2, scale=False).fit(X, Y)
        scores = pls.x_scores_
        df_sc = pd.DataFrame(scores, columns=["Component1", "Component2"])
        df_sc["Class"] = df[label_col].values

        cmap = {cl: c for cl, c in zip(sorted(df_sc["Class"].unique()), [
            "#c3121e","#0348a1","#ffb01c","#027608",
            "#1dace6","#9c5300","#9966cc","#ff4500"
        ])}

        fig = px.scatter(
            df_sc, x="Component1", y="Component2", color="Class",
            template="ggplot2", title=f"{nfeat} Features", color_discrete_map=cmap
        )
        fig.update_traces(marker=dict(size=26, opacity=0.6))
        fig.update_layout(width=600, height=500)
        return fig
# ——————————————
# Run the app
# ——————————————
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)