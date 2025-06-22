import taipy.gui.builder as tgb
from cryptopy.src.taipy_src.helper import dashboard_header

# simulation page
with tgb.Page() as simulation_page:
    dashboard_header()

    # filters
    with tgb.layout():
        tgb.part()
