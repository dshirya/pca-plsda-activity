import re
import os
import random
import warnings
import asyncio
import pandas as pd
import numpy as np

from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget, render_plotly

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go

port = int(os.environ.get("PORT", 8080))
np.random.seed(42)

# ——————————————
# Load your fixed dataset
# ——————————————
df = pd.read_csv("data/features-binary.csv")
label_col = "Class" 
pca_data = "data/elemental-property-list.xlsx"
cluster_data = "data/pauling-data.xlsx"

elements_by_group = {
        "alkali_metals": [
            "Li", "Na", "K", "Rb", "Cs", "Fr"
        ],
        "alkaline_earth_metals": [
            "Be", "Mg", "Ca", "Sr", "Ba", "Ra"
        ],
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
        "metalloids": [
            "B", "Si", "Ge", "As", "Sb", "Te", "Po"
        ],
        "non_metals": [
            "H", "C", "N", "O", "P", "S", "Se"
        ],
        "halogens": [
            "F", "Cl", "Br", "I", "At", "Ts"
        ],
        "noble_gases": [
            "He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"
        ],
        "post_transition_metals": [
            "Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Nh", "Fl", "Mc", "Lv"
        ]
    }

symbol_to_group = {
    el: grp
    for grp, lst in elements_by_group.items()
    for el in lst
}


# ————————————————————————
# Helper functions
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

# Helper: Evaluate a Feature Subset via 5-fold CV
def evaluate_subset(
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: list[str],
        n_components: int = 2,
        scoring: str = "accuracy"
    ) -> float:
    # 1) slice
    X_sub = X[selected_features].copy()

    # 2) drop zero-variance
    zero_var = X_sub.std(axis=0) == 0
    if zero_var.any():
        X_sub = X_sub.loc[:, ~zero_var]

    # 3) still enough for PLS?
    if X_sub.shape[1] < n_components:
        return np.nan

    # 4) scale and clean any NaN/∞
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # 5) one-hot encode
    Y_dummy = pd.get_dummies(y)

    pls = PLSRegression(n_components=n_components, scale=False)
    # 6) suppress the runtime warnings during fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            pls.fit(X_scaled, Y_dummy)
        except Exception:
            return np.nan

    # 7) predict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            y_pred_cont = pls.predict(X_scaled)
        except Exception:
            return np.nan

    # 8) class via argmax
    idxs = np.argmax(y_pred_cont, axis=1)
    preds = [Y_dummy.columns[i] for i in idxs]

    # 9) score
    if scoring == "accuracy":
        return accuracy_score(y, preds)
    else:
        return f1_score(y, preds, average="macro")


def forward_selection_plsda_df(
    numeric_data: pd.DataFrame,
    target_data: pd.Series,
    n_components: int = 2,
    plateau_steps: int = 10,
    init_features: list[str] | None = None,
    scoring: str = "accuracy"
    ) -> pd.DataFrame:
    """
    Greedy forward PLS-DA feature selection with plateau stopping.
    
    - numeric_data: DataFrame of X.
    - target_data: Series of y.
    - n_components: number of PLS components (passed to evaluate_subset).
    - plateau_steps: stop after this many additions without any accuracy gain.
    - init_features: optional list of features to start with.
    
    Assumes evaluate_subset(X, y, feats, n_components, scoring) is defined elsewhere.
    """
    all_feats = list(numeric_data.columns)
    remaining = all_feats.copy()
    selected: list[str] = []
    records: list[dict] = []

    iteration = 0
    best_score = -np.inf
    plateau_count = 0

    # --- INITIAL SEEDING ---
    if init_features:
        for f in init_features:
            if f not in remaining:
                raise ValueError(f"init_features contains '{f}', which is not in numeric_data.")
            remaining.remove(f)
        selected = init_features.copy()
        best_score = evaluate_subset(numeric_data, target_data, selected,
                                     n_components=n_components, scoring=scoring)
        iteration += 1
        records.append({
            'step': iteration,
            'accuracy': best_score,
            'feature_added': ','.join(init_features),
            'total_features': len(selected)
        })
    elif n_components > 1:
        # random seed of size n_components
        seed = random.sample(remaining, n_components)
        for f in seed:
            remaining.remove(f)
        selected = seed.copy()
        best_score = evaluate_subset(numeric_data, target_data, selected,
                                     n_components=n_components, scoring=scoring)
        iteration += 1
        records.append({
            'step': iteration,
            'accuracy': best_score,
            'feature_added': ','.join(seed),
            'total_features': len(selected)
        })

    # --- GREEDY FORWARD WITH PLATEAU STOPPING ---
    while remaining:
        # find best single feature to add
        best_cand, best_cand_score = None, -np.inf
        for feat in random.sample(remaining, len(remaining)):
            trial = selected + [feat]
            if len(trial) < n_components:
                continue
            s = evaluate_subset(numeric_data, target_data, trial,
                                n_components=n_components, scoring=scoring)
            if s > best_cand_score:
                best_cand_score, best_cand = s, feat

        # only add if it doesn’t drop below current best
        if best_cand is None or best_cand_score < best_score:
            break

        # plateau logic
        if best_cand_score == best_score:
            plateau_count += 1
        else:
            plateau_count = 0

        remaining.remove(best_cand)
        selected.append(best_cand)
        iteration += 1
        records.append({
            'step': iteration,
            'accuracy': best_cand_score,
            'feature_added': best_cand,
            'total_features': len(selected)
        })

        # update best_score if improved
        if best_cand_score > best_score:
            best_score = best_cand_score

        # stop if plateau too long
        if plateau_count >= plateau_steps:
            break

    return pd.DataFrame.from_records(records)


def backward_elimination_plsda_df(
    numeric_data: pd.DataFrame,
    target_data: pd.Series,
    min_features: int = 1,
    n_components: int = 2,
    scoring: str = "accuracy"
) -> pd.DataFrame:
    """
    Greedy backward elimination PLS-DA down to `min_features`.
    Returns a DataFrame with columns:
      - step
      - accuracy
      - feature_removed
      - total_features
    """
    # Start with all features
    current = list(numeric_data.columns)
    records: list[dict] = []
    iteration = 0

    # Evaluate the full set
    best_score = evaluate_subset(
        numeric_data, target_data, current,
        n_components=n_components, scoring=scoring
    )
    records.append({
        "step": iteration,
        "accuracy": best_score,
        "feature_removed": "",
        "total_features": len(current)
    })

    # Greedy elimination loop
    while len(current) > min_features:
        best_after = -np.inf
        feat_to_remove = None

        # Try removing each feature
        for feat in random.sample(current, len(current)):
            trial = [f for f in current if f != feat]
            if len(trial) < n_components:
                continue
            s = evaluate_subset(
                numeric_data, target_data, trial,
                n_components=n_components, scoring=scoring
            )
            if s > best_after:
                best_after, feat_to_remove = s, feat

        # If no removal can match or exceed current best, stop
        if feat_to_remove is None or best_after < best_score:
            break

        # Commit the removal
        current.remove(feat_to_remove)
        iteration += 1
        best_score = best_after
        records.append({
            "step": iteration,
            "accuracy": best_score,
            "feature_removed": feat_to_remove,
            "total_features": len(current)
        })

    return pd.DataFrame.from_records(records)

# ——————————————————————————————
# Define your feature groups (display, column) pairs
# ——————————————————————————————
feature_groups_plsda = {
    "index": {
        "label": "Index Features",
        "features": [
            ("A",                 "index_A"),
            ("Index B",                 "index_B"),
            ("Norm A",      "index_A_norm"),
            ("Norm B",      "index_B_norm"),
            ("Max",           "index_max"),
            ("Min",          "index_min"),
            ("Average",           "index_avg"),
        ],
    },
    "atomic_weight": {
        "label": "Atomic Weight",
        "features": [
            ("Weighted A+B",            "atomic_weight_A+B_weighted"),
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
            ("Weighted A+B",    "valencee_total_A+B_weighted"),
            ("Norm Weighted A+B","valencee_total_A+B_weighted_norm"),
        ],
    },
    "unpaired": {
        "label": "Unpaired Electrons",
        "features": [
            ("A",              "unpaired_electrons_A"),
            ("B",              "unpaired_electrons_B"),
            ("A-B",            "unpaired_electrons_A-B"),
            ("A+B",            "unpaired_electrons_A+B"),
            ("Weighted A+B",   "unpaired_electrons_A+B_weighted"),
            ("Norm Weighted A+B","unpaired_electrons_A+B_weighted_norm"),
        ],
    },
    "gilman": {
        "label": "Gilman",
        "features": [
            ("A",                "Gilman_A"),
            ("B",                "Gilman_B"),
            ("A-B",              "Gilman_A-B"),
            ("A+B",              "Gilman_A+B"),
            ("Weighted A+B",     "Gilman_A+B_weighted"),
            ("Norm Weighted A+B", "Gilman_A+B_weighted_norm"),
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
            ("Norm Weighted A+B", "Z_eff_A+B_weighted_norm"),
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
            ("Norm Weighted A+B","ionization_energy_A+B_weighted_norm"),
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
            ("Norm Weighted A+B","CIF_radius_A+B_weighted_norm"),
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
            ("Norm Weighted A+B","Pauling_radius_CN12_A+B_weighted_norm"),
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
            ("Norm Weighted A+B","Pauling_EN_A+B_weighted_norm"),
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
            ("Norm Weighted A+B","Martynov_Batsanov_EN_A+B_weighted_norm"),
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
            ("Norm Weighted A+B","melting_point_K_A+B_weighted_norm"),
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
            ("Norm Weighted A+B","density_A+B_weighted_norm"),
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
            ("Norm Weighted A+B","specific_heat_A+B_weighted_norm"),
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
            ("Norm Weighted A+B","cohesive_energy_A+B_weighted_norm"),
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
            ("Norm Weighted A+B",  "bulk_modulus_A+B_weighted_norm"),
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
    "thermal_and_physical_properties": {
        "label": "Thermal & Physical Properties",
        "features": [
            ("Melting point",          "Melting point, K"),
            ("Boiling point",          "Boiling point, K"),
            ("Density",             "Density,  g/mL"),
            ("Specific heat",      "Specific heat, J/g K"),
            ("Heat of fusion",    "Heat of fusion,  kJ/mol"),
            ("Heat of vaporization",    "Heat of vaporization,  kJ/mol"),
            ("Heat of atomization",     "Heat of atomization,  kJ/mol"),
            ("Thermal conductivity",     "Thermal conductivity, W/m K"),
            ("Cohesive energy",           "Cohesive  energy"),
            ("Bulk modulus",         "Bulk modulus, GPa"),
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
    "thermal_and_physical_properties": {
        "label": "Thermal & Physical Properties",
        "features": [
            ("Melting point",          "Melting point, K"),
            ("Boiling point",          "Boiling point, K"),
            ("Density",             "Density,  g/mL"),
            ("Specific heat",      "Specific heat, J/g K"),
            ("Heat of fusion",    "Heat of fusion,  kJ/mol"),
            ("Heat of vaporization",    "Heat of vaporization,  kJ/mol"),
            ("Heat of atomization",     "Heat of atomization,  kJ/mol"),
            ("Thermal conductivity",     "Thermal conductivity, W/m K"),
            ("Cohesive energy",           "Cohesive  energy"),
            ("Bulk modulus",         "Bulk modulus, GPa"),
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
# at the top of your UI-building code:
default_pls_feats = {
    "Pauling_radius_CN12_A/B",
    "atomic_weight_A+B_weighted"
}

feature_cards_plsda = []
for gid, info in feature_groups_plsda.items():
    # 1) Build feature‐level checkboxes
    checks = []
    for disp, col in info["features"]:
        checks.append(
            ui.input_checkbox(
                f"feat_{make_safe_id(col)}",
                disp,
                value=(col in default_pls_feats),
            )
        )
    group_default = all(col in default_pls_feats for _, col in info["features"])

    header = ui.card_header(
        ui.div(
            ui.input_checkbox(
                f"group_{gid}",
                "",            
                value=group_default,
                width="2em"
            ),
            ui.HTML(f"<b style='margin:0; padding:0;'>{info['label']}</b>"),
            style="display:flex; align-items:center; gap:0; margin:0; padding:0;"
        ),
        style="font-size:1em;"
    )

    feature_cards_plsda.append(
        ui.card(
            header,
            *checks,
            full_screen=False
        )
    )

n_cards = len(feature_cards_plsda)
chunk = -(-n_cards // 4)
cards_pls_col1 = feature_cards_plsda[0:chunk]
cards_pls_col2 = feature_cards_plsda[chunk:2*chunk]
cards_pls_col3 = feature_cards_plsda[2*chunk:3*chunk]
cards_pls_col4 = feature_cards_plsda[3*chunk:]

i = 1   
moved = cards_pls_col3.pop(i)
cards_pls_col4.append(moved)

# pick your defaults here:
default_pca_feats = {
    "Period",
    "Group"
}

feature_cards_pca = []
for gid, info in feature_groups_pca.items():
    # 1) Build feature‐level checkboxes, on if col ∈ default_pca_feats
    checks = []
    for disp, col in info["features"]:
        checks.append(
            ui.input_checkbox(
                f"pca_feat_{make_safe_id(col)}",  # feature‐ID
                disp,                            # display name
                value=(col in default_pca_feats) # default on/off
            )
        )

    # 2) The group box is on only if *all* its children are in the defaults
    group_default = all(col in default_pca_feats for _, col in info["features"])

    # 3) Header with the group checkbox
    header = ui.card_header(
        ui.div(
            ui.input_checkbox(
                f"pca_group_{gid}",   # group‐ID
                "",                   # no label text
                value=group_default,  # default state
                width="2em"
            ),
            ui.HTML(f"<b style='margin:0; padding:0;'>{info['label']}</b>"),
            style="display:flex; align-items:center; gap:0; margin:0; padding:0;"
        ),
        style="font-size:1em;"
    )

    # 4) Assemble the card
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
cards_pca_col1 = feature_cards_pca[:chunk]
cards_pca_col2 = feature_cards_pca[chunk:2*chunk]
cards_pca_col3 = feature_cards_pca[2*chunk:3*chunk]
cards_pca_col4 = feature_cards_pca[3*chunk:]

i = 1   
moved = cards_pca_col3.pop(i)
cards_pca_col4.append(moved)

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
cards_cluster_col1 = feature_cards_cluster[:chunk]
cards_cluster_col2 = feature_cards_cluster[chunk:2*chunk]
cards_cluster_col3 = feature_cards_cluster[2*chunk:3*chunk]
cards_cluster_col4 = feature_cards_cluster[3*chunk:]

i = 1   
moved = cards_cluster_col3.pop(i)
cards_cluster_col4.append(moved)

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
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.div(
                                ui.input_action_button(
                                    "pca_select_all",   
                                    "Select All"),
                                ui.input_action_button(
                                    "pca_deselect_all", 
                                    "Deselect All"),
                            ),
                            ui.row(
                                ui.column(3, *cards_pca_col1),
                                ui.column(3, *cards_pca_col2),
                                ui.column(3, *cards_pca_col3),
                                ui.column(3, *cards_pca_col4),
                            ), 
                            width=875
                        ),
                        ui.div(
                            output_widget("pca_plot"),
                            style="display:flex; justify-content:center; margin-top:12px;"
                        ),
                        ui.row(
                            ui.output_data_frame("pca_contrib_table"),
                            style="display:flex; justify-content:center; margin-top:12px;"
                        ),
                    )
                ),
                ui.nav_panel(
                "Clustering",
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.div(
                                ui.input_action_button(
                                    "clust_select_all",   
                                    "Select All"),
                                ui.input_action_button(
                                    "clust_deselect_all", 
                                    "Deselect All"),
                            ),
                            ui.row(
                                ui.column(3, *cards_cluster_col1),
                                ui.column(3, *cards_cluster_col2),
                                ui.column(3, *cards_cluster_col3),
                                ui.column(3, *cards_cluster_col4),
                            ),
                            width=875
                        ),
                        ui.div(
                            output_widget("clust_plot"),
                            style="display:flex; justify-content:center; margin-top:12px;"
                        ),
                        ui.row(
                            ui.output_data_frame("clust_contrib_table"),
                            style="display:flex; justify-content:center; margin-top:12px;"
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
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.div(
                                ui.input_action_button(
                                    "select_all",   
                                    "Select All"
                                ),
                                ui.input_action_button(
                                    "deselect_all", 
                                    "Deselect All"
                                ),
                                style="display:flex; gap:8px; margin-bottom:12px;"
                            ),
                            ui.row(
                                ui.column(3, *cards_pls_col1),
                                ui.column(3, *cards_pls_col2),
                                ui.column(3, *cards_pls_col3),
                                ui.column(3, *cards_pls_col4),
                            ),
                            width=875
                        ),
                        ui.div(
                            output_widget("pls_plot"),
                            style="display:flex; justify-content:center; margin-top:12px;"
                        ),
                        ui.output_data_frame("contrib_table"),
                        ui.output_data_frame("metrics_table")
                    )
                ),
                ui.nav_panel(
                    "Evaluation",
                        ui.hr(),
                        ui.h3("Forward Feature Selection"),
                            ui.input_action_button("run_forward", "Run"),
                            ui.row(
                                ui.layout_columns(
                                    ui.card(
                                        ui.div(
                                            ui.row(
                                                ui.column(6,
                                                    ui.input_text(
                                                        "init_feat1", 
                                                        "Feature 1", 
                                                        placeholder="e.g. atomic_weight_A"
                                                    )
                                                ),
                                                ui.column(6,
                                                    ui.input_text(
                                                        "init_feat2", 
                                                        "Feature 2", 
                                                        placeholder="e.g. atomic_weight_B"
                                                    )
                                                )
                                            ),
                                            ui.div(
                                                output_widget("forward_perf_plot"), 
                                                style="flex:1;"
                                            ),
                                            ui.output_data_frame("forward_log"),
                                            style="height:800px;"
                                        )
                                    ),   
                                    ui.card(
                                        ui.div(
                                            output_widget("forward_scatter_plot"),
                                            style="flex:1; display:flex; justify-content:center; align-items:center;"
                                        ),
                                        ui.div(
                                            ui.output_ui("forward_slider_ui"),
                                            style="flex:none; display:flex; justify-content:center; margin-top:8px;"
                                        ),   
                                    ),
                                )
                            ),
                        ui.hr(),
                        ui.h3("Backward Feature Selection"),
                            ui.input_action_button("run_backward", "Run"),
                            ui.row(
                                ui.layout_columns(
                                    ui.card(
                                        ui.div(
                                            ui.div(
                                                output_widget("backward_perf_plot"), 
                                                style="flex:1;"
                                            ),
                                            ui.output_data_frame("backward_log"),
                                            ui.output_text_verbatim("backward_final_feats"),
                                            style="height:800px;"
                                        ),
                                        style="display:flex; flex-direction:column; height:500px;"
                                    ),
                                    ui.card(
                                        ui.div(
                                            output_widget("backward_scatter_plot"),
                                            style="display:flex; justify-content:center; margin-top:12px;"
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


# 1) Replace your @reactive.Calc definitions with extended_task:
@reactive.extended_task
async def forward_res(
    numeric_data: pd.DataFrame,
    target_data: pd.Series,
    init_features: list[str] | None
) -> pd.DataFrame:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        forward_selection_plsda_df,
        numeric_data,
        target_data,
        2,          # n_components
        10,         # plateau_steps
        init_features,
        "accuracy"
    )

@reactive.extended_task
async def backward_res(
    numeric_data: pd.DataFrame,
    target_data: pd.Series
) -> pd.DataFrame:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        backward_elimination_plsda_df,
        numeric_data,
        target_data,
        2,          # min_features
        2,          # n_components
        "accuracy"
    )

# ——————————————
# Server logic
# ——————————————
def server(input, output, session):
    
    prev_group_states = {
      gid: all(col in default_pls_feats for _, col in info["features"])
      for gid, info in feature_groups_plsda.items()
    }
    prev_group_states_pca = {
      gid: all(col in default_pls_feats for _, col in info["features"])
      for gid, info in feature_groups_pca.items()
    }

    # ——————————————
    # PCA calculation
    # ——————————————

    @reactive.Effect
    @reactive.event(input.pca_select_all)
    def select_all():
        for gid, info in feature_groups_pca.items():
            ui.update_checkbox(f"pca_group_{gid}", value=True, session=session)
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"pca_feat_{fid}", value=True, session=session)

    @reactive.Effect
    @reactive.event(input.pca_deselect_all)
    def deselect_all():
        for gid, info in feature_groups_pca.items():
            ui.update_checkbox(f"pca_group_{gid}", value=False, session=session)
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"pca_feat_{fid}", value=False, session=session)


    # When a group toggles, toggle its features
    for gid, info in feature_groups_pca.items():
        @reactive.Effect
        @reactive.event(getattr(input, f"pca_group_{gid}"))
        def _grp_toggle(gid=gid, info=info):
            new_state = getattr(input, f"pca_group_{gid}")()
            old_state = prev_group_states_pca[gid]
            if new_state != old_state:
                prev_group_states_pca[gid] = new_state
                for _, col in info["features"]:
                    fid = make_safe_id(col)
                    ui.update_checkbox(f"pca_feat_{fid}", value=new_state, session=session)

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

    @render_plotly
    def pca_plot():
        res = pca_res()
        if res is None:
            fig = go.Figure()
            fig.update_layout(
                title="No features selected or no data.",
                template="ggplot2",
                width=800,
                height=800,
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
        fig.update_traces(marker=dict(
                                    size=26, 
                                    opacity=0.6))
        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=False,
            font=dict(size=16),
            autosize=True,
            width=800,
            height=800,
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
    def clust_select_all():
        for gid, info in feature_groups_cluster.items():
            ui.update_checkbox(f"clust_group_{gid}", value=True, session=session)
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"clust_feat_{fid}", value=True, session=session)

    @reactive.Effect
    @reactive.event(input.clust_deselect_all)
    def clust_deselect_all():
        for gid, info in feature_groups_cluster.items():
            ui.update_checkbox(f"clust_group_{gid}", value=False, session=session)
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"clust_feat_{fid}", value=False, session=session)

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
            formula = row["Formula"]
            a, b = split_formula(formula)
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
                "struct": row["Structure type"],
                "formula": formula
            })

        return {
            "dfp": dfp,
            "pc1_pct": pc1_pct,
            "pc2_pct": pc2_pct,
            "contribution": contrib_df,
            "links": links
        }
    # 4) render the clustering plot
    @render_plotly
    def clust_plot():
        res = clust_res()
        if not res:
            fig = go.Figure()
            fig.update_layout(
                title="No data",
                width=800, 
                height=830
            )
            return fig

        dfp = res["dfp"]
        fig = px.scatter(
            dfp, 
            x="PC1", 
            y="PC2", 
            text="Symbol", 
            template="ggplot2", 
            width=800, 
            height=830
        )
        fig.update_traces(
            marker=dict(
                color="green", 
                size=26, 
                opacity=0.6
            )
        )

        # colour map for your three structure types
        cmap = {
            "CsCl": "#c3121e",
            "NaCl": "#0348a1",
            "ZnS":  "#ffb01c",  
        }

        seen = set()
        for link in res["links"]:
            struct = link["struct"]
            col = cmap.get(struct, "#888888")

            # 1) line trace, grouped with its markers
            fig.add_trace(go.Scatter(
                x=link["x"],
                y=link["y"],
                mode="lines",
                line=dict(color=col, width=1),
                opacity=0.4,
                legendgroup=struct,
                showlegend=False,
                hoverinfo="none"
            ))

            # 2) midpoint marker, same group
            show_leg = struct not in seen
            fig.add_trace(go.Scatter(
                x=[link["mid_x"]],
                y=[link["mid_y"]],
                mode="markers",
                marker=dict(color=col, size=10, opacity=0.6),
                name=struct,
                legendgroup=struct,
                showlegend=show_leg,
                hovertext=[link["formula"]],
                hovertemplate="%{hovertext}"
            ))
            seen.add(struct)

        fig.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            font=dict(size=16),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            width=800, height=830
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
    def plsda_select_all():
        for gid, info in feature_groups_plsda.items():
            ui.update_checkbox(f"group_{gid}", value=True, session=session)
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"feat_{fid}", value=True, session=session)

    @reactive.Effect
    @reactive.event(input.deselect_all)
    def plsda_deselect_all():
        for gid, info in feature_groups_plsda.items():
            ui.update_checkbox(f"group_{gid}", value=False, session=session)
            for _, col in info["features"]:
                fid = make_safe_id(col)
                ui.update_checkbox(f"feat_{fid}", value=False, session=session)

    for gid, info in feature_groups_plsda.items():
        @reactive.Effect
        @reactive.event(getattr(input, f"group_{gid}"))
        def _grp_toggle(gid=gid, info=info):
            new_state = getattr(input, f"group_{gid}")()
            old_state = prev_group_states[gid]
            # only run if it really changed
            if new_state != old_state:
                prev_group_states[gid] = new_state
                # propagate to all children
                for _, col in info["features"]:
                    fid = make_safe_id(col)
                    ui.update_checkbox(f"feat_{fid}", value=new_state, session=session)

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

        if len(valid) < 2:
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
                autosize=False,
                width=800,
                height=830
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
        fig.update_traces(
            marker=dict(
                size=26, 
                opacity=0.8
            )
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(size=18),
            width=800,
            height=830,
            showlegend=True,

            legend_orientation="h",
            legend_x=0.5,         
            legend_xanchor="center",
            legend_y=1.02,         
            legend_yanchor="bottom",
            legend_title_side="top center"
        )

        fig.update_yaxes(
            scaleanchor="x", 
            scaleratio=1
        )
        fig.update_xaxes(
            constrain="domain"
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
    # 2) Forward Feature Selection
    # ——————————————————————————————————————
    @reactive.Effect
    @reactive.event(input.run_forward)
    def _start_forward():
        numeric = df.drop(columns=[label_col])
        target  = df[label_col]
        f1 = input.init_feat1().strip()
        f2 = input.init_feat2().strip()
        init = [f1, f2] if (f1 and f2) else None
        forward_res(numeric, target, init)

    @render_plotly
    def forward_perf_plot():
        if input.run_forward() < 1 or forward_res.result() is None:
            fig = go.Figure()
            fig.update_layout(
                title="Waiting to run forward selection…",
                template="ggplot2"
            )
            return fig

        df_steps = forward_res.result()
        fig = go.Figure(go.Scatter(
            x=df_steps['total_features'],
            y=df_steps['accuracy'],
            mode="lines+markers"
        ))
        fig.update_layout(
            title="Forward Selection Accuracy",
            xaxis_title="Number of Features",
            yaxis_title="Accuracy",
            template="ggplot2",
            xaxis_range=[
                df_steps['total_features'].min() - 1,
                df_steps['total_features'].max() + 1
            ],
            height=300,
        )
        return fig

    @render.data_frame
    def forward_log():
        if input.run_forward() < 1 or forward_res.result() is None:
            return pd.DataFrame(columns=[
                'step', 'total_features', 'feature_added', 'accuracy'
            ])
        return forward_res.result()

    @render.ui
    def forward_slider_ui():
        # before any run (or still running), lock at 2
        if input.run_forward() < 1 or forward_res.result() is None:
            return ui.input_slider(
                "forward_step", 
                "Features",
                min=2, max=2, value=2, step=1
            )
        df_steps = forward_res.result()
        counts = df_steps["total_features"].astype(int)
        low, high = counts.min(), counts.max()
        return ui.input_slider(
            "forward_step",
            "Features",
            min=low, max=high,
            value=low, step=1,
        )

    @render_plotly
    def forward_scatter_plot():
        if input.run_forward() < 1 or forward_res.result() is None:
            fig = go.Figure()
            fig.update_layout(
                title="…waiting for forward selection…",
                width=600, height=550
            )
            return fig

        df_steps = forward_res.result()
        selected_n = input.forward_step()
        mask = df_steps["total_features"] == selected_n
        if not mask.any():
            fig = go.Figure()
            fig.update_layout(
                title=f"No record for {selected_n} features.",
                width=600, height=550
            )
            return fig

        idx = df_steps.index[mask][0]
        added_lists = df_steps.loc[:idx, "feature_added"].str.split(",")
        sel_feats = [feat for sub in added_lists for feat in sub]

        X_scaled = StandardScaler().fit_transform(df[sel_feats])
        Y_dummy  = pd.get_dummies(df[label_col])
        pls = PLSRegression(n_components=2, scale=False).fit(X_scaled, Y_dummy)
        scores = pls.x_scores_

        df_sc = pd.DataFrame(scores, columns=["Component1","Component2"])
        df_sc["Class"] = df[label_col].values

        cmap = {
            cl: c for cl,c in zip(
                sorted(df_sc["Class"].unique()),
                ["#c3121e","#0348a1","#ffb01c","#027608",
                 "#1dace6","#9c5300","#9966cc","#ff4500"]
            )
        }
        fig = px.scatter(
            df_sc,
            x="Component1", y="Component2",
            color="Class",
            template="ggplot2",
            title=f"{selected_n} Features",
            color_discrete_map=cmap
        )
        fig.update_traces(marker=dict(size=26, opacity=0.6))
        fig.update_layout(width=600, height=550)
        return fig


    # ─────────────────────────────────────────────────────────────────────────────
    # 3) Backward Elimination (backgrounded)
    # ─────────────────────────────────────────────────────────────────────────────
    @reactive.Effect
    @reactive.event(input.run_backward)
    def _start_backward():
        numeric = df.drop(columns=[label_col])
        target  = df[label_col]
        backward_res(numeric, target)

    @render_plotly
    def backward_perf_plot():
        if input.run_backward() < 1 or backward_res.result() is None:
            fig = go.Figure()
            fig.update_layout(
                title="Waiting to run backward selection…",
                template="ggplot2"
            )
            return fig

        df_steps = backward_res.result()
        fig = go.Figure(go.Scatter(
            x=df_steps['total_features'],
            y=df_steps['accuracy'],
            mode="lines+markers"
        ))
        low  = df_steps['total_features'].min() - 1
        high = df_steps['total_features'].max() + 1
        fig.update_layout(
            title="Backward Elimination Accuracy",
            xaxis_title="Number of Features",
            yaxis_title="Accuracy",
            template="ggplot2", height=300
        )
        fig.update_xaxes(autorange=False, range=[high, low], tickmode='linear')
        return fig

    @render.data_frame
    def backward_log():
        if input.run_backward() < 1 or backward_res.result() is None:
            return pd.DataFrame(columns=[
                'step', 'accuracy', 'feature_removed', 'total_features'
            ])
        return backward_res.result()

    @render.ui
    def backward_slider_ui():
        if input.run_backward() < 1 or backward_res.result() is None:
            # lock slider at full set size until done
            full_n = len(df.columns) - 1
            return ui.input_slider(
                "backward_step", "Features",
                min=2, max=full_n, value=full_n, step=-1
            )
        df_steps = backward_res.result()
        counts = df_steps['total_features'].tolist()
        low, high = min(counts), max(counts)
        return ui.input_slider(
            "backward_step", "Features",
            min=low, max=high, value=high, step=-1
        )

    @render_plotly
    def backward_scatter_plot():
        if input.run_backward() < 1 or backward_res.result() is None:
            fig = go.Figure()
            fig.update_layout(
                title="…waiting for backward selection…",
                width=600, height=550
            )
            return fig

        df_steps = backward_res.result()
        target_n = input.backward_step()
        current_feats = list(df.drop(columns=[label_col]).columns)

        for _, row in df_steps.sort_values('step').iterrows():
            if row['feature_removed'] and len(current_feats) > target_n:
                current_feats.remove(row['feature_removed'])
            if len(current_feats) == target_n:
                break

        X = StandardScaler().fit_transform(df[current_feats])
        Y = pd.get_dummies(df[label_col])
        pls = PLSRegression(n_components=2, scale=False).fit(X, Y)
        scores = pls.x_scores_

        df_sc = pd.DataFrame(scores, columns=["Component1","Component2"])
        df_sc["Class"] = df[label_col].values

        cmap = {
            cl: c for cl,c in zip(
                sorted(df_sc["Class"].unique()),
                ["#c3121e","#0348a1","#ffb01c","#027608",
                 "#1dace6","#9c5300","#9966cc","#ff4500"]
            )
        }
        fig = px.scatter(
            df_sc, x="Component1", y="Component2",
            color="Class",
            template="ggplot2",
            title=f"{len(current_feats)} Features",
            color_discrete_map=cmap
        )
        fig.update_traces(marker=dict(size=26, opacity=0.6))
        fig.update_layout(width=600, height=550)
        return fig
# ——————————————
# Run the app
# ——————————————
app = App(app_ui, server)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
