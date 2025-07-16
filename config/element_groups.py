# Element classification by groups
ELEMENTS_BY_GROUP = {
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

# Create mapping from symbol to group
SYMBOL_TO_GROUP = {
    el: grp
    for grp, lst in ELEMENTS_BY_GROUP.items()
    for el in lst
} 