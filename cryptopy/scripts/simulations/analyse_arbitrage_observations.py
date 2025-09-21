import pandas as pd
import matplotlib.pyplot as plt
from pandas import json_normalize

pd.set_option("display.max_columns", None)

JSON_FILE = (
    r"C:\Users\thoma\PycharmProjects\CryptoDashboard\data\arbitrage"
    r"\statistical_arbitrage.jsonl"
)


def load_records(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def tidy_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # ---------------- unpack `key` ----------------
    def unpack_key(row):
        k0, k1, k2, k3 = (row + [None] * 4)[:4]
        return pd.Series(
            {
                "arb_type": k0,
                "coins": k1,
                "exchanges": k2,
                "path": k3,
            }
        )

    df = pd.concat([df.drop(columns="key"), df["key"].apply(unpack_key)], axis=1)

    # ---------------- unpack `meta` ----------------
    df = pd.concat([df.drop(columns="meta"), json_normalize(df["meta"])], axis=1)

    # ---------------- timestamps ----------------
    for col in ["timestamp", "start_ts", "last_ts", "end_ts"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")

    # ------------- convenient aliases -------------
    # keep original but add a shorter alias so plotting code is nicer
    if "lifetime_seconds" in df.columns:
        df["duration"] = df["lifetime_seconds"]

    df = df.loc[:, ~df.columns.duplicated()]
    return df


# ----------------------------------------------------------------------
# 2 .  Visualisations
# ----------------------------------------------------------------------
def plot_stats(df: pd.DataFrame) -> None:
    """Three bar charts (counts, avg lifetime, avg profit)."""

    # ------------- split views -------------
    df_final = df[df["event"] == "finalise"]  # only completed opportunities
    df_updates = df  # all rows (updates + finalise)

    g_final = df_final.groupby("arb_type")
    g_updates = df_updates.groupby("arb_type")

    # ---------- 1. number completed ----------
    g_final.size().plot(kind="bar", title="Completed opportunities")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

    # ---------- 2. mean lifetime ----------
    g_final["duration"].mean().plot(
        kind="bar", title="Mean lifetime of completed opps (s)"
    )
    plt.ylabel("seconds")
    plt.tight_layout()
    plt.show()

    # ---------- 3. mean realised profit ----------
    g_updates["avg_profit"].mean().plot(
        kind="bar", title="Avg profit during opportunity (£)"
    )
    plt.ylabel("£ (per £100 notional)")
    plt.tight_layout()
    plt.show()

    # ---------- 4. histogram of profit ----------
    if "avg_profit" in df_final.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(df_final["avg_profit"], bins=50, color="skyblue", edgecolor="black")
        plt.title("Distribution of Realised Profits")
        plt.xlabel("Profit (£ per 100 notional)")
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.6)
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# 3 .  Main
# ----------------------------------------------------------------------
def main() -> None:
    raw = load_records(JSON_FILE)
    tidy = tidy_frame(raw)

    if "timestamp" in tidy.columns:
        latest_date = tidy["timestamp"].max()
        one_year_ago = latest_date - pd.Timedelta(days=365)
        tidy = tidy[tidy["timestamp"] >= one_year_ago]

    print("\n=== headline metrics (completed only) ===\n")
    print(
        tidy[tidy["event"] == "finalise"]
        .groupby("arb_type")
        .agg(
            count=("duration", "size"),
            mean_duration=("duration", "mean"),
            mean_profit=("avg_profit", "mean"),
            max_profit=("max_profit", "max"),
            min_profit=("min_profit", "min"),
        )
        .round(3)
    )

    plot_stats(tidy)


if __name__ == "__main__":
    main()
