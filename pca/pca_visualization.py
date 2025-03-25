import os
import matplotlib.pyplot as plt
from utils import appearance as props
from utils.preprocess import parse_formula

def save_plot(structure, coord_sheet_name, ax, folder=props.plot_folder) -> None:
    """
    Saves the plotted periodic table.
    
    Args:
        structure (str): Name of the structure.
        coord_sheet_name (str): Name of the coordinate sheet used.
        ax: Matplotlib axis object.
        folder (str): Directory to save the plot.
    """
    os.makedirs(folder, exist_ok=True)
    structure_clean = structure.replace(" ", "_")
    coord_sheet_clean = coord_sheet_name.replace(" ", "_")
    base_filename = f"{structure_clean}_{coord_sheet_clean}{props.file_extension}"
    file_path = os.path.join(folder, base_filename)

    counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(folder, f"{structure_clean}_{coord_sheet_clean}_{counter}{props.file_extension}")
        counter += 1

    plt.savefig(file_path, dpi=props.dpi, bbox_inches=props.bbox_inches)
    print(f"Plot saved as {file_path}")

def get_figure_size(coord_df):
    """Compute figure size based on element coordinates."""
    x_min, x_max = coord_df["x"].min(), coord_df["x"].max()
    y_min, y_max = coord_df["y"].min(), coord_df["y"].max()
    fig_width = (x_max - x_min) * 0.8
    fig_height = (y_max - y_min) * 0.8
    return fig_width, fig_height

def plot_elements(ax, coord_df):
    """Plot the PCA background elements as circles with labels."""
    for _, row in coord_df.iterrows():
        x, y, symbol = row['x'], row['y'], row['Symbol']
        ax.add_patch(plt.Circle((x, y), props.circle_radius, fill=None, edgecolor='black', lw=props.shape_linewidth))
        ax.text(x, y, symbol, ha='center', va='center', fontsize=props.text_fontsize_circle, zorder=5)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def build_marker_lookup(marker_df):
    """Precompute a dictionary mapping compound formula to its (x, y) marker coordinate."""
    return {row["Formula"]: (row["x"], row["y"]) for _, row in marker_df.iterrows()}

def build_composition_lookup(data_df):
    """Precompute a dictionary mapping each compound formula to its parsed composition."""
    composition_lookup = {}
    for _, compound in data_df.iterrows():
        formula = compound["Formula"]
        if formula not in composition_lookup:
            composition_lookup[formula] = parse_formula(formula)
    return composition_lookup

def display_data(ax, data_df, marker_lookup, element_dict, composition_lookup):
    """
    Highlights compounds by drawing their average markers, connecting them to their constituent elements,
    and adding shrinking circles on the elements.
    """
    # Assign colors based on unique structure types.
    structures = sorted(data_df["Structure type"].unique())
    structure_colors = {structure: props.colors[i % len(props.colors)] for i, structure in enumerate(structures)}
    added_labels = set()
    
    # These dictionaries keep track of drawn patches for repeated elements.
    circle_counts = {}
    applied_colors = {}
    structure_markers = {}
    marker_index = 0

    for _, compound in data_df.iterrows():
        formula = compound["Formula"]
        structure = compound["Structure type"]

        # Retrieve compound marker coordinates from the precomputed lookup.
        if formula not in marker_lookup:
            print(f"Warning: Marker for formula {formula} not found.")
            continue
        center_x, center_y = marker_lookup[formula]

        # Set a unique marker style per structure.
        if structure not in structure_markers:
            structure_markers[structure] = props.marker_types[marker_index]
            marker_index += 1
        marker = structure_markers[structure]
        color = structure_colors[structure]

        # Plot the compound's average coordinate marker.
        if structure not in added_labels:
            ax.scatter(center_x, center_y, edgecolors=color, facecolors='None', label=structure,
                       zorder=4, s=props.marker_size, marker=marker, alpha=1, linewidths=4)
            added_labels.add(structure)
        else:
            ax.scatter(center_x, center_y, edgecolors=color, facecolors='None',
                       zorder=4, s=props.marker_size, marker=marker, alpha=1, linewidths=4)

        # Get the compound's elemental composition from the lookup.
        composition = composition_lookup.get(formula, {})

        # Draw connections and element highlights.
        for element, count in composition.items():
            if element in element_dict:
                x, y = element_dict[element]
                # Draw a light connection line from the compound marker to the element.
                ax.plot([center_x, x], [center_y, y], color=color, linestyle='-', zorder=2, alpha=0.1)

                # Track how many times an element is highlighted.
                if (x, y) not in circle_counts:
                    circle_counts[(x, y)] = 0
                if (x, y) not in applied_colors:
                    applied_colors[(x, y)] = set()

                # Skip if this color was already applied at this element coordinate.
                if color in applied_colors[(x, y)]:
                    continue

                count_patch = circle_counts[(x, y)]
                shrink_factor = props.shrink_factor_circle * count_patch
                size = props.circle_size - shrink_factor
                ax.add_patch(plt.Circle((x, y), size, fill=False, edgecolor=color,
                                        zorder=4, linewidth=props.linewidth_circle, alpha=0.8))
                applied_colors[(x, y)].add(color)
                circle_counts[(x, y)] += 1
            else:
                print(f"Warning: Element {element} not found for formula {formula}")

    plt.legend(**props.legend_props)
    first_structure = data_df.iloc[0]["Structure type"] if not data_df.empty else "default"
    save_plot(first_structure, 'PCA_plot', ax)

def PCA_plot(coord_df, data_df, marker_df, structure_type) -> plt.Axes:
    """
    Main function to plot the PCA view of elements, highlight compound markers,
    and draw connections from compound average coordinates to their constituent elements.
    
    Args:
        coord_df (pd.DataFrame): DataFrame with element coordinates (columns: 'Symbol', 'x', 'y').
        data_df (pd.DataFrame): DataFrame with compound data (must include 'Formula' and 'Structure type').
        marker_df (pd.DataFrame): DataFrame with compound average coordinates (columns: 'Formula', 'x', 'y').
    
    Returns:
        ax: Matplotlib axis object.
    """
    # Build a simple dictionary mapping element symbols to their coordinates.
    element_dict = {row['Symbol']: (row['x'], row['y']) for _, row in coord_df.iterrows()}
    figsize_x, figsize_y = get_figure_size(coord_df)
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    plot_elements(ax, coord_df)
    
    # Precompute lookups to avoid repeated DataFrame filtering and parsing.
    marker_lookup = build_marker_lookup(marker_df)
    composition_lookup = build_composition_lookup(data_df)

    display_data(ax, data_df, marker_lookup, element_dict, composition_lookup)
    return ax