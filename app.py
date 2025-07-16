"""
PCA-PLSDA Application
A modular Shiny application for chemical data analysis using PCA and PLS-DA.
"""

from shiny import App

# Import configuration
from config.settings import PORT

# Import UI
from ui.layout import create_app_ui

# Import server logic (we'll create a main server function)
from server.main import create_server


def main():
    """Main application entry point."""
    app_ui = create_app_ui()
    app_server = create_server()
    
    app = App(app_ui, app_server)
    return app

# Create the app instance
app = main()

if __name__ == "__main__":
    # Run the app when the script is executed directly
    app.run(host="0.0.0.0", port=PORT) 