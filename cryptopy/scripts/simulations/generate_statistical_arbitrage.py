# stat_arb_finder.py
# -----------------------------------------------------------
import json
import glob
from pathlib import Path
from typing import List, Tuple
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm


# ---------- 1. data ingest -------------------------------------------------
def load_prices(data_dir: str, price_column: str = "close") -> pd.DataFrame:
    frames = []
    for fp in Path(data_dir).glob("*.csv"):
        sym = fp.stem.split(".")[0]  # e.g. BTC_USD
        df = (
            pd.read_csv(fp, parse_dates=["datetime"])
            .set_index("datetime")[price_column]
            .rename(sym)
        )
        # If datetime already tz‑aware this does nothing, else localise to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        frames.append(df)

    wide = pd.concat(frames, axis=1).sort_index()
    wide = wide.ffill().dropna(how="all")  # forward‑fill then drop all‑NaN rows
    return wide


# ---------- 2. pair selection ---------------------------------------------
def find_best_pair(
    wide: pd.DataFrame, p_threshold: float = 0.05
) -> Tuple[str, str, float]:
    best_p = 1.0
    best_pair: tuple[str, str] | None = None
    cols = wide.columns

    for col_i, col_j in itertools.combinations(cols, 2):
        # Grab the two price series, then CLEAN & ALIGN them
        x_ser, y_ser = prepare_pair(wide[col_i], wide[col_j])

        # coint returns (stat, p‑value, crit_vals) – we need p‑value
        p_val = coint(x_ser, y_ser)[1]
        if p_val < p_threshold and p_val < best_p:
            best_p, best_pair = p_val, (col_i, col_j)

    if best_pair is None:
        raise ValueError(f"No cointegrated pair found (p < {p_threshold}).")
    return (*best_pair, best_p)


def prepare_pair(
    s1: pd.Series,
    s2: pd.Series,
    min_obs: int = 100,
) -> tuple[pd.Series, pd.Series]:
    """
    Align two Series on their common index, drop NaN / inf rows,
    ensure we still have enough data.
    """
    x, y = s1.align(s2, join="inner")  # keep only shared timestamps
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < min_obs:
        raise ValueError("Not enough overlapping observations after cleaning")

    return df["x"], df["y"]


# ---------- 3. hedge ratio & spread ---------------------------------------
def hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """
    OLS β of y = α + β x  .  Any row that still has NaN/inf is dropped
    so statsmodels will not raise MissingDataError.
    """
    df = pd.DataFrame({"y": y, "x": x}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 30:  # protect against tiny sample
        raise ValueError("too few observations for OLS")

    x_const = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], x_const).fit()
    return model.params[1]


def compute_spread(
    wide: pd.DataFrame, x_sym: str, y_sym: str, roll: int = 30
) -> pd.DataFrame:
    """
    Align, clean and return the spread/z‑score table.
    Raises ValueError if not enough clean data for the rolling window.
    """
    x_ser, y_ser = prepare_pair(wide[x_sym], wide[y_sym])  # already cleaned
    if len(x_ser) < roll + 5:
        raise ValueError("series shorter than rolling window")

    beta = hedge_ratio(x_ser, y_ser)
    s = x_ser - beta * y_ser
    mu = s.rolling(roll).mean()
    sigma = s.rolling(roll).std()
    z = (s - mu) / sigma
    return pd.DataFrame(
        {
            "x": x_ser,
            "y": y_ser,
            "spread": s,
            "mu": mu,
            "sigma": sigma,
            "z": z,
            "beta": beta,
        }
    )


# ---------- 4. signal extraction ------------------------------------------
def extract_trades(
    spread_df: pd.DataFrame, entry_z: float = 2.0, trade_size=100_000
) -> pd.DataFrame:
    in_trade = False
    records = []

    for ts, row in spread_df.dropna().iterrows():
        z = row.z
        if not in_trade and abs(z) >= entry_z:
            # ---- open ----
            in_trade = True
            direction = -1 if z > 0 else 1  # +1 means long X/short Y
            open_ts = ts
            open_px_x = row.x
            open_px_y = row.y
            beta = row.beta
            qty_x = trade_size / open_px_x  # amount of X bought/sold
            qty_y = beta * qty_x  # hedge ratio amount of Y
            open_z = z
        elif in_trade:
            # look for zero-cross
            if (direction == -1 and z <= 0) or (direction == 1 and z >= 0):
                close_px_x = row.x
                close_px_y = row.y

                # Profit: X leg
                pnl_x = qty_x * (close_px_x - open_px_x) * direction
                # Profit: Y leg (opposite direction)
                pnl_y = qty_y * (open_px_y - close_px_y) * direction

                profit = pnl_x + pnl_y

                hours = (ts - open_ts).total_seconds() / 3600
                records.append(
                    {
                        "entry_ts": open_ts,
                        "exit_ts": ts,
                        "direction": direction,
                        "hours_held": hours,
                        "profit": profit,
                        "entry_z": open_z,
                        "exit_z": z,
                        "qty_x": qty_x,
                        "qty_y": qty_y,
                        "entry_px_x": open_px_x,
                        "entry_px_y": open_px_y,
                        "exit_px_x": close_px_x,
                        "exit_px_y": close_px_y,
                        "beta": beta,
                        "notional": trade_size,
                    }
                )
                in_trade = False

    return pd.DataFrame.from_records(records)


# def extract_trades(
#     spread_df: pd.DataFrame, entry_z: float = 2.0, trade_size=100_000
# ) -> pd.DataFrame:
#     """
#     Returns one row per completed trade:
#     entry_ts, exit_ts, direction (+1 long‑spread, ‑1 short‑spread),
#     hours_held, profit(pips of spread), entry_z, exit_z
#     """
#     in_trade = False
#     direction = 0
#     records = []
#
#     for ts, row in spread_df.dropna().iterrows():
#         z = row.z
#         if not in_trade and abs(z) >= entry_z:
#             # ---- open ----
#             in_trade = True
#             direction = -1 if z > 0 else 1  # +1 means buy x / sell y
#             open_ts = ts
#             open_s = row.spread
#             open_z = z
#         elif in_trade:
#             # look for zero‑cross
#             if (direction == -1 and z <= 0) or (direction == 1 and z >= 0):
#                 # ---- close ----
#                 hours = (ts - open_ts).total_seconds() / 3600
#                 # profit = (open_s - row.spread) * direction * trade_size  # Δ spread
#                 profit_pct = ((open_s - row.spread) / abs(open_s)) * direction
#                 profit = profit_pct * trade_size
#                 records.append(
#                     {
#                         "entry_ts": open_ts,
#                         "exit_ts": ts,
#                         "direction": direction,
#                         "hours_held": hours,
#                         "profit": profit,
#                         "entry_z": open_z,
#                         "exit_z": z,
#                     }
#                 )
#                 in_trade = False
#
#     return pd.DataFrame.from_records(records)


# ---------- 5. JSON export -----------------------------------------------
def to_json_records(
    trades: pd.DataFrame, x_sym: str, y_sym: str, beta: float, save_path: str
):
    """
    Serialise each trade in the same 'update/finalise' style
    you use for simple / triangular logs.
    """
    out_lines: List[str] = []
    for _, r in trades.iterrows():
        line = {
            "event": "finalise",
            "key": [
                "statistical",
                [x_sym, y_sym],
                [],  # exchanges not used offline
                [[x_sym, y_sym]],
            ],
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "start_ts": r.entry_ts.value / 1e9,
            "end_ts": r.exit_ts.value / 1e9,
            "lifetime_seconds": r.hours_held * 3600,
            "observations": None,
            "last_profit": r.profit,
            "avg_profit": r.profit,
            "max_profit": r.profit,
            "min_profit": r.profit,
            "profit_stdev": 0.0,
            "meta": {
                "total_profit": r.profit,
                "coins_used": [x_sym, y_sym],
                "beta": beta,
                "entry_z": r.entry_z,
                "exit_z": r.exit_z,
            },
            "reason": "completed",
            "sharpe_like": None,
        }
        out_lines.append(json.dumps(line, default=str))

    Path(save_path).write_text("\n".join(out_lines))
    print(f"[+] wrote {len(out_lines)} stat‑arb trades to {save_path}")


# ---------- 6. glue it together -------------------------------------------
def run_stat_arb(folder: str, out_json: str, roll_window: int = 30):
    prices = load_prices(folder)
    x_sym, y_sym, p = find_best_pair(prices)
    print(f"Best cointegrated pair: {x_sym}/{y_sym}  (p={p:.3g})")

    spread_df = compute_spread(prices, x_sym, y_sym, roll_window)
    trades = extract_trades(spread_df)
    if trades.empty:
        print("No completed trades found in history.")
        return

    print(trades[["entry_ts", "exit_ts", "hours_held", "profit"]])
    beta = spread_df.beta.iloc[0]
    to_json_records(trades, x_sym, y_sym, beta, out_json)


def run_stat_arb_all(
    folder: str,
    out_json: str,
    roll_window: int = 30,
    p_threshold: float = 0.05,
    min_obs: int = 200,
    entry_z: float = 2.0,
):
    prices = load_prices(folder)
    kept = 0
    total_trades = 0

    for col_i, col_j in itertools.combinations(prices.columns, 2):
        try:
            x_ser, y_ser = prepare_pair(prices[col_i], prices[col_j], min_obs=min_obs)
        except ValueError:  # not enough overlap
            continue

        p_val = coint(x_ser, y_ser)[1]
        if p_val >= p_threshold:
            continue  # not cointegrated enough

        spread_df = compute_spread(prices, col_i, col_j, roll_window)
        trades = extract_trades(spread_df, entry_z)
        if trades.empty:
            continue

        beta = hedge_ratio(spread_df["x"], spread_df["y"])
        json_lines = []
        for _, r in trades.iterrows():
            json_lines.append(
                {
                    "event": "finalise",
                    "key": [
                        "statistical",
                        [col_i, col_j],
                        [],  # offline, so no exchanges
                        [[col_i, col_j]],
                    ],
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "start_ts": r.entry_ts.value / 1e9,
                    "end_ts": r.exit_ts.value / 1e9,
                    "lifetime_seconds": r.hours_held * 3600,
                    "observations": None,
                    "last_profit": r.profit,
                    "avg_profit": r.profit,
                    "max_profit": r.profit,
                    "min_profit": r.profit,
                    "profit_stdev": 0.0,
                    "meta": {
                        "total_profit": r.profit,
                        "coins_used": [col_i, col_j],
                        "beta": beta,
                        "entry_z": r.entry_z,
                        "exit_z": r.exit_z,
                        "p_value": p_val,
                    },
                    "reason": "completed",
                    "sharpe_like": None,
                }
            )

        append_json_lines(json_lines, out_json)
        kept += 1
        total_trades += len(json_lines)
        print(
            f"[+] {col_i}/{col_j}  p={p_val:.3g}  trades={len(json_lines)}  "
            f"(saved → {out_json})"
        )

    if kept == 0:
        print("No cointegrated pairs with completed trades were found.")
    else:
        print(
            f"\nFinished.  Saved {total_trades} trades "
            f"from {kept} cointegrated pairs."
        )


def append_json_lines(lines: list[dict], save_path: str) -> None:
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    with save_p.open("a", encoding="utf-8") as fh:
        for d in lines:
            fh.write(json.dumps(d, default=str) + "\n")


# ----------------- example usage ------------------------------------------
if __name__ == "__main__":
    DATA_DIR = r"C:\Users\thoma\PycharmProjects\CryptoDashboard\data\historical_data\Kraken_long_history"  # folder with *.csv files
    OUT_FILE = r"C:\Users\thoma\PycharmProjects\CryptoDashboard\data\arbitrage\statistical_arbitrage.jsonl"  # one JSON per line

    run_stat_arb_all(DATA_DIR, OUT_FILE, roll_window=30)
