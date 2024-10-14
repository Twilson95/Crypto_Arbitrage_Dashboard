import numpy as np


def format_amount(value, unit=""):
    if value == 0:
        return f"0 {unit}"
    elif abs(value) >= 10000:
        return f"{abs(value):,.0f} {unit}"
    elif abs(value) >= 1000:
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


def generate_log_marks():
    marks = {}
    for i in range(0, 6):  # From 10^0 to 10^5
        for j in range(1, 10):  # 1 to 9 to create intermediate marks
            value = j * 10**i
            log_position = np.log10(value)
            marks[log_position] = "{:.0f}".format(value)
    # Adding the last point explicitly to cover the 10^5 range
    marks[5] = "{:.0f}".format(10**5)
    return marks
