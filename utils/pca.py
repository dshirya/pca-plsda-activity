import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    fig.update_traces(marker=dict(size=26, symbol="circle", opacity=0.6))
    fig.update_layout(
        width=1200,
        height=1200,
        xaxis=dict(scaleanchor='y', scaleratio=1),
        yaxis=dict(scaleanchor='x', scaleratio=1),
        showlegend=False,
        font=dict(size=18)
    )
    return fig