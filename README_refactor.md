# PCA-PLSDA Application - Refactored Structure

This document describes the new modular file structure for the PCA-PLSDA application.

## 📁 File Structure Overview

```
pca-plsda-activity/
├── app_new.py              # New main entry point
├── app.py                  # Original monolithic file (keep for reference)
├── config/                 # Configuration and data definitions
│   ├── __init__.py
│   ├── settings.py         # App settings, data paths, defaults
│   ├── element_groups.py   # Element classifications
│   └── feature_groups.py   # Feature group definitions
├── utils/                  # Pure functions and algorithms
│   ├── __init__.py
│   ├── helpers.py          # Utility functions
│   ├── algorithms.py       # PLS-DA and evaluation algorithms
│   └── data_processing.py  # Data loading and preprocessing
├── ui/                     # User interface components
│   ├── __init__.py
│   ├── layout.py           # Main UI layout
│   ├── components.py       # Reusable UI components
│   └── cards.py            # Feature selection cards
├── server/                 # Server logic organized by functionality
│   ├── __init__.py
│   ├── main.py             # Main server assembly
│   ├── pca.py              # PCA-related server logic
│   ├── plsda.py            # PLS-DA-related server logic
│   ├── clustering.py       # Clustering-related server logic
│   └── evaluation.py       # Feature selection evaluation logic
├── data/                   # (existing data files)
└── requirements.txt        # (existing)
```

## 🎯 Benefits of This Structure

### 1. **Separation of Concerns**
- **Config**: All settings, defaults, and data definitions in one place
- **Utils**: Pure functions that can be easily tested and reused
- **UI**: Clean separation of layout, components, and cards
- **Server**: Server logic organized by functionality

### 2. **Maintainability**
- Each file has a single, clear responsibility
- Easy to find and modify specific features
- Reduces coupling between different parts of the application

### 3. **Reusability**
- UI components can be reused across different parts of the app
- Utility functions are pure and easily testable
- Configuration is centralized and easy to modify

### 4. **Scalability**
- Easy to add new analysis types or features
- Clear patterns for extending functionality
- Modular structure supports team development

## 🚀 How to Use

### Running the Application

```bash
# Using the new modular structure
python app_new.py

# Or still use the original (for comparison)
python app.py
```

### Making Changes

#### Adding New Features
1. **New Analysis Type**: Add to `config/feature_groups.py` and create corresponding server module
2. **New UI Component**: Add to `ui/components.py`
3. **New Algorithm**: Add to `utils/algorithms.py`

#### Modifying Existing Features
1. **Change UI Layout**: Edit `ui/layout.py`
2. **Modify Analysis Logic**: Edit the appropriate server module
3. **Update Configuration**: Edit files in `config/`

#### Adding New Data Sources
1. Add data path to `config/settings.py`
2. Add loading function to `utils/data_processing.py`
3. Update relevant server modules

## 📋 Module Descriptions

### Config Modules

#### `config/settings.py`
- Application configuration (ports, paths, defaults)
- UI configuration (sizes, colors)
- Algorithm parameters
- Color schemes

#### `config/element_groups.py`
- Element classifications by group
- Mapping from element symbols to groups

#### `config/feature_groups.py`
- Feature group definitions for PLS-DA, PCA, and clustering
- Display names and column mappings

### Utils Modules

#### `utils/helpers.py`
- `make_safe_id()`: Convert names to safe HTML IDs
- `get_class_color_map()`: Create color mappings
- `split_formula()`: Parse chemical formulas
- `clean_array()`: Handle NaN and infinite values

#### `utils/algorithms.py`
- `evaluate_subset()`: Evaluate feature subsets via PLS-DA
- `forward_selection_plsda_df()`: Forward feature selection
- `backward_elimination_plsda_df()`: Backward feature elimination

#### `utils/data_processing.py`
- `load_*_data()`: Data loading functions
- `prepare_plsda_data()`: Data preparation for PLS-DA
- `filter_valid_features()`: Feature validation
- `remove_zero_variance_features()`: Data cleaning

### UI Modules

#### `ui/layout.py`
- `create_app_ui()`: Main application layout
- Complete UI structure with all tabs and panels

#### `ui/components.py`
- `create_sidebar_with_cards()`: Reusable sidebar component
- `create_plot_output()`: Standardized plot outputs
- `create_evaluation_card()`: Evaluation panel cards

#### `ui/cards.py`
- `create_feature_cards()`: Generate feature selection cards
- `distribute_cards_to_columns()`: Balance card layout
- Specific functions for each analysis type

### Server Modules

#### `server/main.py`
- Combines all server modules
- Main server function assembly

#### `server/pca.py`
- PCA calculation and visualization
- Element mapping functionality
- Feature selection for PCA

#### `server/plsda.py` (to be created)
- PLS-DA analysis and visualization
- Classification metrics
- Feature contribution analysis

#### `server/clustering.py` (to be created)
- Clustering visualization
- Structure type mapping

#### `server/evaluation.py` (to be created)
- Forward and backward feature selection
- Performance tracking and visualization

## 🔧 Implementation Status

### ✅ Completed
- [x] Config package with settings and feature groups
- [x] Utils package with helpers, algorithms, and data processing
- [x] UI package with layout, components, and cards
- [x] Server package structure and main assembly
- [x] PCA server module
- [x] New main app.py file

### 🚧 Remaining Work
- [ ] Complete server modules (plsda.py, clustering.py, evaluation.py)
- [ ] Test the modular application
- [ ] Verify all functionality works as expected
- [ ] Add error handling and validation
- [ ] Update documentation

## 📝 Migration Guide

### From Original to Modular Structure

1. **Configuration Changes**: Update imports to use `config.*` modules
2. **Function Calls**: Update function calls to use new module structure
3. **Server Logic**: Each analysis type now has its own server module
4. **UI Components**: Use new reusable components from `ui.components`

### Testing the Migration

1. Run both versions side-by-side to compare functionality
2. Test each analysis type (PCA, PLS-DA, clustering, evaluation)
3. Verify all UI interactions work correctly
4. Check that data loading and processing work as expected

## 🎨 Code Style and Patterns

### Naming Conventions
- **Files**: snake_case (e.g., `feature_groups.py`)
- **Functions**: snake_case (e.g., `create_app_ui()`)
- **Classes**: PascalCase (if any)
- **Constants**: UPPER_CASE (e.g., `DEFAULT_PLS_FEATURES`)

### Import Patterns
```python
# Config imports
from config.settings import PORT, UI_CONFIG
from config.feature_groups import FEATURE_GROUPS_PLSDA

# Utils imports  
from utils.helpers import make_safe_id
from utils.algorithms import evaluate_subset

# UI imports
from ui.components import create_sidebar_with_cards
```

### Error Handling
- Use try-except blocks for external dependencies
- Provide fallback values for configuration
- Return None or empty DataFrames for invalid states

This modular structure makes the application much easier to maintain, extend, and debug. Each component has a clear purpose and can be modified independently. 