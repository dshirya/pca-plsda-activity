import os
import pandas as pd

def load_and_prepare_data(filepath):
    """
    Loads an Excel file, drops columns with missing values,
    and selects numeric columns (dropping any rows with missing values).
    Optionally returns the 'Symbol' column if available.
    """
    try:
        data = pd.read_excel(filepath)
        data_clean = data.dropna(axis=1)
        numeric_data = data_clean.select_dtypes(include=['float64', 'int64']).copy()
        numeric_data.dropna(inplace=True)
        symbol_data = data_clean.loc[numeric_data.index, 'Symbol'] if 'Symbol' in data_clean.columns else None
        return data_clean, numeric_data, symbol_data
    except Exception as e:
        print("Error loading file:", e)
        return None, None, None

def get_unique_filename(filepath):
    """
    Returns a unique filepath by appending '_1', '_2', etc. 
    if the base filepath already exists.
    """
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath
    while os.path.exists(new_filepath):
        new_filepath = f"{base}_{counter}{ext}"
        counter += 1
    return new_filepath