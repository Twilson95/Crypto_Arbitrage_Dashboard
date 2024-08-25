import numpy as np


def format_amount(value, unit=""):
    if value == 0:
        return f"0 {unit}"
    elif abs(value) >= 10000:
        return f"{abs(value):,.1f} {unit}"
    elif abs(value) >= 1:
        return f"{abs(value):,.2f} {unit}"
    else:
        return f"{abs(value):.3g} {unit}"


def round_to_significant_figures(value, sig_figs):
    if value == 0:
        return 0
    else:
        return round(value, sig_figs - int(np.floor(np.log10(abs(value)))) - 1)
