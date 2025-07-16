"""Main UI layout combining all components."""

from shiny import ui
from shinywidgets import output_widget
from .cards import create_plsda_feature_cards, create_pca_feature_cards, create_cluster_feature_cards
from .components import create_select_deselect_buttons, distribute_cards_to_columns


def create_app_ui():
    """Create the main application UI."""
    
    # Create feature cards for each analysis type
    plsda_cards = create_plsda_feature_cards()
    pca_cards = create_pca_feature_cards()
    cluster_cards = create_cluster_feature_cards()
    
    # Distribute cards across columns
    plsda_cols = distribute_cards_to_columns(plsda_cards, num_columns=4)
    pca_cols = distribute_cards_to_columns(pca_cards, num_columns=4)
    cluster_cols = distribute_cards_to_columns(cluster_cards, num_columns=4)
    
    # Build main UI
    return ui.page_fluid(
        ui.navset_card_tab(
            ui.nav_panel(
                "PCA",
                ui.page_navbar(
                    ui.nav_panel(
                        "Element Mapping",
                        ui.layout_sidebar(
                            ui.sidebar(
                                create_select_deselect_buttons("pca"),
                                ui.row(*[
                                    ui.column(3, *col_cards) 
                                    for col_cards in pca_cols
                                ]),
                                width=875
                            ),
                            ui.div(
                                # PCA plot output
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
                                create_select_deselect_buttons("clust"),
                                ui.row(*[
                                    ui.column(3, *col_cards) 
                                    for col_cards in cluster_cols
                                ]),
                                width=875
                            ),
                            ui.div(
                                # Clustering plot output
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
                                create_select_deselect_buttons(""),  # PLS-DA uses no prefix
                                ui.row(*[
                                    ui.column(3, *col_cards) 
                                    for col_cards in plsda_cols
                                ]),
                                width=875
                            ),
                            ui.div(
                                # PLS-DA plot output
                                output_widget("pls_plot"),
                                style="display:flex; justify-content:center; margin-top:12px;"
                            ),
                            ui.output_data_frame("contrib_table"),
                            ui.output_data_frame("metrics_table"),
                            ui.output_data_frame("vip_table")
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