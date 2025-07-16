import re
import numpy as np


def make_safe_id(name: str) -> str:
    """
    Convert a name to a safe HTML/CSS ID by replacing special characters.
    
    Args:
        name: The name to convert
        
    Returns:
        A safe ID string
    """
    # First give / and + unique replacements
    s = name.replace("/", "_slash_").replace("+", "_plus_")
    # Then replace any non-alphanumeric-or-underscore with underscore
    safe = re.sub(r'\W+', '_', s)
    # Ensure it doesn't start with a digit
    if re.match(r'^\d', safe):
        safe = 'f_' + safe
    return safe


def get_class_color_map(classes, color_scheme):
    """
    Create a color mapping for classes.
    
    Args:
        classes: List of unique class labels
        color_scheme: Dictionary mapping indices to colors
        
    Returns:
        Dictionary mapping class labels to colors
    """
    return {
        cl: color_scheme.get(i, "#888888")
        for i, cl in enumerate(sorted(classes))
    }


def split_formula(formula: str) -> list[str]:
    """
    Split a chemical formula into element symbols.
    
    Args:
        formula: Chemical formula string
        
    Returns:
        List of element symbols
    """
    return re.findall(r"[A-Z][a-z]?", formula)


def safe_division(a, b, default=0.0):
    """
    Safely divide two numbers, returning default if division by zero.
    
    Args:
        a: Numerator
        b: Denominator  
        default: Value to return if b is zero
        
    Returns:
        Result of a/b or default if b is zero
    """
    return a / b if b != 0 else default


def clean_array(arr, nan_value=0.0, inf_value=0.0):
    """
    Clean an array by replacing NaN and infinite values.
    
    Args:
        arr: Input array
        nan_value: Value to replace NaN with
        inf_value: Value to replace infinite values with
        
    Returns:
        Cleaned array
    """
    return np.nan_to_num(arr, nan=nan_value, posinf=inf_value, neginf=inf_value) 