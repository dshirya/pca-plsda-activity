"""Reusable UI components for the application."""

from shiny import ui
from shinywidgets import output_widget


def create_select_deselect_buttons(prefix=""):
    """
    Create Select All and Deselect All buttons.
    
    Args:
        prefix: Prefix for the button IDs (e.g., "pca", "clust", or "" for PLS-DA)
        
    Returns:
        UI div with buttons
    """
    if prefix:
        select_id = f"{prefix}_select_all"
        deselect_id = f"{prefix}_deselect_all"
    else:
        # PLS-DA uses no prefix
        select_id = "select_all"
        deselect_id = "deselect_all"
    
    return ui.div(
        ui.input_action_button(select_id, "Select All"),
        ui.input_action_button(deselect_id, "Deselect All"),
        style="display:flex; gap:8px; margin-bottom:12px;"
    )


def distribute_cards_to_columns(cards, num_columns=4):
    """
    Distribute cards evenly across columns.
    
    Args:
        cards: List of UI cards
        num_columns: Number of columns to distribute across
        
    Returns:
        List of lists, each containing cards for one column
    """
    if not cards:
        return [[] for _ in range(num_columns)]
    
    n_cards = len(cards)
    chunk_size = -(-n_cards // num_columns)  # Ceiling division
    
    columns = []
    for i in range(num_columns):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, n_cards)
        if start_idx < n_cards:
            columns.append(cards[start_idx:end_idx])
        else:
            columns.append([])
    
    # Manual adjustment like in original app.py
    if num_columns == 4 and len(columns[2]) > 0:
        try:
            moved = columns[2].pop(1)  # Move second item from column 3 to column 4
            columns[3].append(moved)
        except (IndexError, AttributeError):
            pass  # If adjustment fails, continue with current distribution
    
    return columns


def create_sidebar_with_cards(card_columns, select_all_id, deselect_all_id):
    """
    Create a sidebar with select/deselect buttons and feature cards.
    
    Args:
        card_columns: List of card lists for each column
        select_all_id: ID for select all button
        deselect_all_id: ID for deselect all button
        
    Returns:
        UI sidebar
    """
    return ui.sidebar(
        ui.div(
            ui.input_action_button(select_all_id, "Select All"),
            ui.input_action_button(deselect_all_id, "Deselect All"),
            style="display:flex; gap:8px; margin-bottom:12px;"
        ),
        ui.row(*[
            ui.column(3, *col_cards) 
            for col_cards in card_columns
        ]),
        width=875
    )


def create_plot_output(output_id):
    """
    Create a plot output div with centering.
    
    Args:
        output_id: ID for the plot output
        
    Returns:
        UI div with plot output
    """
    return ui.div(
        output_widget(output_id),
        style="display:flex; justify-content:center; margin-top:12px;"
    )


def create_data_table_output(output_id):
    """
    Create a data table output with centering.
    
    Args:
        output_id: ID for the data table output
        
    Returns:
        UI row with data table output
    """
    return ui.row(
        ui.output_data_frame(output_id),
        style="display:flex; justify-content:center; margin-top:12px;"
    )


def create_evaluation_card(title, plot_id, table_id, additional_content=None):
    """
    Create an evaluation card with plot and table.
    
    Args:
        title: Card title
        plot_id: ID for the plot output
        table_id: ID for the table output
        additional_content: Optional additional UI content
        
    Returns:
        UI card
    """
    content = [
        ui.div(
            output_widget(plot_id), 
            style="flex:1;"
        ),
        ui.output_data_frame(table_id),
    ]
    
    if additional_content:
        content.insert(1, additional_content)
    
    return ui.card(
        ui.div(*content, style="height:800px;")
    )


def create_scatter_plot_card(plot_id, slider_ui_id):
    """
    Create a scatter plot card with slider.
    
    Args:
        plot_id: ID for the scatter plot output
        slider_ui_id: ID for the slider UI output
        
    Returns:
        UI card
    """
    return ui.card(
        ui.div(
            output_widget(plot_id),
            style="flex:1; display:flex; justify-content:center; align-items:center;"
        ),
        ui.div(
            ui.output_ui(slider_ui_id),
            style="flex:none; display:flex; justify-content:center; margin-top:8px;"
        ),   
    )


def create_action_button_with_spacing(button_id, label):
    """
    Create an action button with spacing.
    
    Args:
        button_id: ID for the button
        label: Button label
        
    Returns:
        UI action button
    """
    return ui.input_action_button(button_id, label) 