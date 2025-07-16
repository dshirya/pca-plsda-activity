"""
Main server module that combines all server logic.
"""

def create_server():
    """
    Create the main server function by combining all server modules.
    
    Returns:
        Server function for the Shiny app
    """
    
    def server(input, output, session):
        # Import and initialize all server modules
        from .pca import setup_pca_server
        from .plsda import setup_plsda_server  
        from .clustering import setup_clustering_server
        from .evaluation import setup_evaluation_server
        
        # Set up each module's server logic
        setup_pca_server(input, output, session)
        setup_plsda_server(input, output, session)
        setup_clustering_server(input, output, session)
        setup_evaluation_server(input, output, session)
    
    return server 