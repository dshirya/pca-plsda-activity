'''

Here you can adjust appearance of the plot and periodic table

'''
multiplier = 1.2


# Marker types for different structures
marker_types = ['o', 's', 'D', '^', '*', 'P', '2', '8', 'X', 'h']
marker_size = 60 * multiplier  # Default size for scatter markers

# Colors for different structures
colors = [  
    "#c3121e",  # Sangre (195, 18, 30)
    "#0348a1",  # Neptune (3, 72, 161)
    "#ffb01c",  # Pumpkin (255, 176, 28)
    "#027608",  # Clover (2, 118, 8)
    "#1dace6",  # Cerulean (29, 172, 214)
    "#9c5300",  # Cocoa (156, 83, 0)
    "#9966cc",  # Amethyst (153, 102, 204) 
]


shape_linewidth = 2  # Line width for shapes (circles/rectangles)

# Plot
circle_radius = 0.3 * multiplier # Radius for circles in periodic table (default 0.3)
circle_size = circle_radius - 0.04 # Default size for color circles (default 0.26)
shrink_factor_circle = 0.054  # How much each circle shrinks with additional layers (better not to change) (default 0.054)
text_fontsize_circle = 16 * multiplier  # Font size for text in circles (default 18)
linewidth_circle = 2 

# Legend properties
legend_props = {
    "fontsize": 18 * multiplier, ##24
    "loc": 'upper center',
    "frameon": False,
    "framealpha": 1,
    "edgecolor": 'black',
    "markerscale": 1,
    "ncol": 3
}

# Axis properties
axis_visibility = False  # Whether to show axis lines and ticks
aspect_ratio = 'equal'  # Aspect ratio for plots

# Plot saving properties
plot_folder = "plots"
file_extension = ".png"
dpi = 600
bbox_inches = 'tight'
