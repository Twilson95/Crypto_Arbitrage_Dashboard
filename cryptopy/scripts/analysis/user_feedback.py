import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
xlsx_path = Path(
    r"C:\Users\thoma\Documents\Learning\Masters\Year 2\Project\dashboard feedback.xlsx"
)  # Change to your actual file path

# --- LOAD ---
try:
    df = pd.read_excel(xlsx_path, header=[0, 1])  # two-row header
except ValueError:
    df = pd.read_excel(xlsx_path)  # fallback to one-row header


# --- SELECT Q2 COLUMNS ---
def get_col(df, top, sub):
    if isinstance(df.columns, pd.MultiIndex):
        return pd.to_numeric(df[(top, sub)], errors="coerce")
    else:
        col = [
            c
            for c in df.columns
            if top.lower() in str(c).lower() and sub.lower() in str(c).lower()
        ][0]
        return pd.to_numeric(df[col], errors="coerce")


simple = get_col(df, "Q2: Time to identify trade (sec)", "Simple").dropna()
tri = get_col(df, "Q2: Time to identify trade (sec)", "Triangular").dropna()
stat = get_col(df, "Q2: Time to identify trade (sec)", "Statistical").dropna()

# --- PLOT ---
plt.figure(figsize=(8, 6))
plt.boxplot(
    [simple, tri, stat], labels=["Simple", "Triangular", "Statistical"], showmeans=True
)
plt.title("Time to Identify Trade by Arbitrage Type")
plt.ylabel("Seconds")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(
    r"C:\Users\thoma\Documents\Learning\Masters\Year 2\Project\feedback_box_and_whisker_plots.png"
)
plt.show()
