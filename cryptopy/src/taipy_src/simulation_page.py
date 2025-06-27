import taipy.gui.builder as tgb

from cryptopy.src.taipy_src.callbacks import on_simulation_init
from cryptopy.src.taipy_src.helper import dashboard_header

# simulation page
with tgb.Page(on_init=on_simulation_init) as simulation_page:
    dashboard_header()

    # filters
    with tgb.layout():
        tgb.part()
