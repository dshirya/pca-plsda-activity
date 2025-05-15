import re
import numpy as np
import pandas as pd

def parse_formula(formula: str) -> dict:
    """
    Parse a chemical formula into its element components and counts.
    
    Uses the regex pattern: ([A-Z][a-z]*)([0-9]*\.?[0-9]*)
    
    Parameters:
        formula (str): Chemical formula, e.g., 'AgMg' or 'C6H12O6'.
        
    Returns:
        dict: A dictionary with element symbols as keys and their counts as values.
              If no count is provided for an element, count defaults to 1.
    """
    pattern = r'([A-Z][a-z]*)([0-9]*\.?[0-9]*)'
    matches = re.findall(pattern, formula)
    elements = {}
    for elem, count in matches:
        # If count is empty, default to 1
        if count == '':
            count = 1
        else:
            # Convert count to float if it contains a dot, else to int
            count = float(count) if '.' in count else int(count)
        # If the element appears more than once, sum the counts
        elements[elem] = elements.get(elem, 0) + count
    return elements

def calculate_average_coordinate(data_df: pd.DataFrame, coords_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the weighted average coordinate (x, y) for each compound.
    
    For each compound in data_df, parse its formula using parse_formula.
    For each element in the compound, multiply the coordinate (from coords_df)
    by its count in the compound. The average coordinate is the sum of these
    products divided by the total number of atoms.
    
    Parameters:
        data_df (pd.DataFrame): DataFrame containing at least a 'Formula' column.
        coords_df (pd.DataFrame): DataFrame containing columns 'Symbol', 'x', and 'y'
                                  for the element coordinates.
    
    Returns:
        pd.DataFrame: A DataFrame with columns ['Formula', 'avg_x', 'avg_y'] for each compound.
    """
    result_rows = []
    
    for _, row in data_df.iterrows():
        formula = row['Formula']
        composition = parse_formula(formula)
        sum_x = 0.0
        sum_y = 0.0
        total_atoms = 0
        
        for element, count in composition.items():
            # Find the row in coords_df corresponding to the element symbol
            element_data = coords_df[coords_df['Symbol'] == element]
            if not element_data.empty:
                x = element_data.iloc[0]['x']
                y = element_data.iloc[0]['y']
                sum_x += x * count
                sum_y += y * count
                total_atoms += count
            else:
                print(f"Warning: Element {element} not found in coordinates DataFrame")
        
        if total_atoms > 0:
            avg_x = sum_x / total_atoms
            avg_y = sum_y / total_atoms
        else:
            avg_x, avg_y = np.nan, np.nan
        
        result_rows.append({'Formula': formula, 'x': avg_x, 'y': avg_y})
    
    return pd.DataFrame(result_rows)