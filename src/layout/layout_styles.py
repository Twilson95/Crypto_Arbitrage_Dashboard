container_style = {
    "display": "flex",
    "flex-direction": "column",
    "height": "100vh",  # Use full viewport height
    # "margin": "5px",
    "overflow-x": "hidden",
}

filter_container_style = {
    "display": "flex",
    "justify-content": "space-between",
    "align-items": "center",
    "width": "100%",
    "padding": "10px",
    "outline": "2px solid white",
    # "outline": "2px solid red",
}
filter_style = {
    "flex": "1 1 auto",
    "padding-left": "1%",
    "padding-right": "1%",
    # "outline": "2px solid green",
}
grid_container_style = {
    "flex": "1",  # Take the remaining space
    "display": "flex",
    "flex-direction": "column",
    # "justify-content": "center",
    "align-items": "center",
    "width": "100%",
    "padding": "5px",
    # "outline": "2px solid blue",
}
header_style = {
    # "padding": "5px",
    # "height": "80px",
    "align-items": "center",
    "justify-content": "center",
}
grid_row_style = {
    "width": "100%",
    "display": "flex",
    "flex": "1",
    "justify-content": "stretch",
    "align-items": "stretch",
    "margin": "0px",
    # "outline": "2px solid orange",
}
grid_element_style = {
    "flex": "1",  # flex-grow, flex-shrink, flex-basis
    "display": "flex",
    "flex-direction": "column",  # Make sure the flex direction is column
    "height": "100%",  # Set a fixed height to prevent shrinking
    "max-width": "50%",
    "padding": "2.5px",
    "overflow": "hidden",
    # "outline": "2px solid yellow",
}
style_table = {
    "flex": "1",
    "height": "290px",
    "overflowY": "scroll",  # Enable vertical scrolling
    "overflowX": "auto",  # Enable horizontal scrolling if needed
    "width": "100%",  # Ensure the table takes full width of the container
    "padding-left": "0",
    "padding-right": "0",
}
style_cell = {
    "height": "auto",  # Adjust cell height automatically
    "maxWidth": "200px",
    "width": "auto",  # Allow width to adjust automatically
    "whiteSpace": "normal",  # Ensure content wraps within cells
    "overflow": "hidden",
    "textOverflow": "ellipsis",
}
