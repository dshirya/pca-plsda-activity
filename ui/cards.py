from shiny import ui
from utils.helpers import make_safe_id
from config.feature_groups import FEATURE_GROUPS_PLSDA, FEATURE_GROUPS_PCA
from config.settings import DEFAULT_PLS_FEATURES, DEFAULT_PCA_FEATURES


def create_plsda_feature_cards():
    """Create PLS-DA feature selection cards (exactly like original app.py)."""
    cards = []
    
    for gid, info in FEATURE_GROUPS_PLSDA.items():
        # Build feature-level checkboxes
        checks = []
        for disp, col in info["features"]:
            checks.append(
                ui.input_checkbox(
                    f"feat_{make_safe_id(col)}",
                    disp,
                    value=(col in DEFAULT_PLS_FEATURES),
                )
            )
        
        # Check if all features in this group are selected by default
        group_default = all(col in DEFAULT_PLS_FEATURES for _, col in info["features"])

        # Create header with group checkbox (exactly like original)
        header = ui.card_header(
            ui.div(
                ui.input_checkbox(
                    f"group_{gid}",  # PLS-DA uses just "group_{gid}"
                    "",            
                    value=group_default,
                    width="2em"
                ),
                ui.HTML(f"<b style='margin:0; padding:0;'>{info['label']}</b>"),
                style="display:flex; align-items:center; gap:0; margin:0; padding:0;"
            ),
            style="font-size:1em;"
        )

        cards.append(
            ui.card(
                header,
                *checks,
                full_screen=False
            )
        )
    
    return cards


def create_pca_feature_cards():
    """Create PCA feature selection cards (exactly like original app.py)."""
    cards = []
    
    for gid, info in FEATURE_GROUPS_PCA.items():
        # Build feature-level checkboxes
        checks = []
        for disp, col in info["features"]:
            checks.append(
                ui.input_checkbox(
                    f"pca_feat_{make_safe_id(col)}",  # PCA uses "pca_feat_"
                    disp,
                    value=(col in DEFAULT_PCA_FEATURES),
                )
            )
        
        # Check if all features in this group are selected by default
        group_default = all(col in DEFAULT_PCA_FEATURES for _, col in info["features"])

        # Create header with group checkbox (exactly like original)
        header = ui.card_header(
            ui.div(
                ui.input_checkbox(
                    f"pca_group_{gid}",  # PCA uses "pca_group_{gid}"
                    "",            
                    value=group_default,
                    width="2em"
                ),
                ui.HTML(f"<b style='margin:0; padding:0;'>{info['label']}</b>"),
                style="display:flex; align-items:center; gap:0; margin:0; padding:0;"
            ),
            style="font-size:1em;"
        )

        cards.append(
            ui.card(
                header,
                *checks,
                full_screen=False
            )
        )
    
    return cards


def create_cluster_feature_cards():
    """Create clustering feature selection cards (exactly like original app.py)."""
    cards = []
    
    for gid, info in FEATURE_GROUPS_PCA.items():
        # Build feature-level checkboxes (all default to True like original)
        checks = []
        for disp, col in info["features"]:
            checks.append(
                ui.input_checkbox(
                    f"clust_feat_{make_safe_id(col)}",  # Clustering uses "clust_feat_"
                    disp,
                    value=True  # All default to True in original
                )
            )

        # Create header with group checkbox (exactly like original)
        header = ui.card_header(
            ui.div(
                ui.input_checkbox(
                    f"clust_group_{gid}",  # Clustering uses "clust_group_{gid}"
                    "",
                    value=True,  # All default to True in original
                    width="2em",
                ),
                ui.HTML(f"<b style='margin:0; padding:0;'>{info['label']}</b>"),
                style="display:flex; align-items:center; gap:0; margin:0; padding:0;"
            ),
            style="font-size:1em;"
        )
        
        cards.append(
            ui.card(
                header, 
                *checks, 
                full_screen=False
            )
        )
    
    return cards


def distribute_cards_to_columns(cards, n_columns=4):
    """
    Distribute cards into roughly equal columns.
    
    Args:
        cards: List of cards
        n_columns: Number of columns to create
        
    Returns:
        List of card lists for each column
    """
    n_cards = len(cards)
    chunk = -(-n_cards // n_columns)  # Ceiling division
    
    columns = []
    for i in range(n_columns):
        start_idx = i * chunk
        end_idx = min((i + 1) * chunk, n_cards)
        columns.append(cards[start_idx:end_idx])
    
    # Balance columns if needed (move one card from col3 to col4)
    if len(columns) >= 4 and len(columns[2]) > 1:
        moved = columns[2].pop(1)
        columns[3].append(moved)
    
    return columns 