from matplotlib import pyplot as plt
import numpy as np
import json
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 20,  # base font size
        "axes.labelsize": 20,  # axis labels
        "axes.titlesize": 20,  # titles
        "xtick.labelsize": 20,  # x-axis tick labels
        "ytick.labelsize": 20,  # y-axis tick labels
        "legend.fontsize": 20,  # legend
    }
)


def load_order_book(filename):
    with open(filename, "r") as f:
        return json.load(f)


path = r"C:\Users\thoma\PycharmProjects\CryptoDashboard\data\order_book\order_book_btc.json"
order_book = load_order_book(path)
asks = sorted(order_book["asks"], key=lambda x: x[0])
bids = sorted(order_book["bids"], key=lambda x: -x[0])

trade_sizes = np.linspace(0, 500_000, 500)  # USD amounts to test
taker_fee_pct = 0.001  # 0.1%
gas_fee = 1.3  # fixed USD cost per transfer


def calc_vwap_slippage(order_book_side, amount_usd, is_buy=True):
    remaining = amount_usd
    cost = 0
    total_qty = 0
    best_price = order_book_side[0][0]

    for price, liquidity in order_book_side:
        if remaining <= 0:
            break
        qty = min(liquidity, remaining)
        cost += price * qty
        total_qty += qty
        remaining -= qty

    vwap = cost / total_qty if total_qty > 0 else best_price
    if is_buy:
        slippage_pct = (vwap - best_price) / best_price * 100
    else:
        slippage_pct = (best_price - vwap) / best_price * 100
    return slippage_pct, vwap


slippages_total = []
fees_total = []
net_losses = []

slippages_total_pct = []
fees_total_pct = []
net_losses_pct = []

for size in trade_sizes:
    # Buy leg (into asks)
    slippage_buy, vwap_buy = calc_vwap_slippage(asks, size, is_buy=True)
    # Sell leg (into bids)
    slippage_sell, vwap_sell = calc_vwap_slippage(bids, size, is_buy=False)

    # Combined slippage = buy + sell
    combined_slippage = slippage_buy + slippage_sell
    slippages_total_pct.append(combined_slippage)

    # Fees as %
    fees = (taker_fee_pct * size * 2) + gas_fee
    fees_pct = (fees / size) * 100
    fees_total_pct.append(fees_pct)

    # Net loss (USD)
    gross_loss = (vwap_buy - vwap_sell) * (
        size / ((vwap_buy + vwap_sell) / 2)
    )  # approx USD impact
    net_loss = gross_loss + fees
    net_loss_pct = (net_loss / size) * 100
    net_losses_pct.append(net_loss_pct)

min_idx = np.argmin(net_losses_pct)
optimal_trade_size = trade_sizes[min_idx]
optimal_net_loss = net_losses_pct[min_idx]

# Plot with annotation for min net loss
plt.figure(figsize=(10, 6))
plt.plot(trade_sizes, slippages_total_pct, label="Slippage (%)", color="blue")
plt.plot(trade_sizes, fees_total_pct, label="Fees (%)", color="red")
plt.plot(trade_sizes, net_losses_pct, label="Net Loss (%)", color="green")
plt.scatter(optimal_trade_size, optimal_net_loss, color="black", zorder=5)

plt.annotate(
    f"Min Net Loss: {optimal_net_loss:.2f}%\nTrade Size: ${optimal_trade_size:,.0f}",
    (optimal_trade_size, optimal_net_loss),
    textcoords="offset points",
    xytext=(2, -45),
    ha="left",
    fontsize=18,
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
)
plt.xlabel("Trade Size (USD)")
plt.ylabel("Percentage of Trade Size (%)")
plt.legend(loc="best")
plt.tight_layout()
# Save with higher DPI for visibility
plt.savefig(
    r"C:\Users\thoma\Documents\Learning\Masters\Year 2\Project\trade_amount_v_fees_and_slippage.png",
    dpi=300,  # <-- High DPI for print / zoom
    bbox_inches="tight",
)


order_book_path = r"C:\Users\thoma\PycharmProjects\CryptoDashboard\data\order_book\order_book_btc.json"
order_book = load_order_book(order_book_path)
asks = sorted(order_book["asks"], key=lambda x: x[0])
bids = sorted(order_book["bids"], key=lambda x: -x[0])


def calc_vwap(order_book_side, amount_usd):
    remaining = amount_usd
    cost = 0
    total_qty = 0
    for price, liquidity in order_book_side:
        if remaining <= 0:
            break
        qty = min(liquidity, remaining)
        cost += price * qty
        total_qty += qty
        remaining -= qty
    return cost / total_qty if total_qty > 0 else order_book_side[0][0]


def calc_slippage_cost(asks, bids, amount_usd, number_of_legs=2):
    """
    Estimate slippage cost for multi-leg arbitrage.
    For triangular arbitrage (3 legs), assume extra leg uses same bid-side liquidity as sells.
    """
    # First leg: buy at asks
    vwap_buy = calc_vwap(asks, amount_usd)
    # Second leg: sell at bids
    vwap_sell1 = calc_vwap(bids, amount_usd)

    # Additional leg(s): treat as sells using same bid liquidity
    extra_leg_cost = 0
    for _ in range(number_of_legs - 2):
        vwap_sell_extra = calc_vwap(bids, amount_usd)
        extra_leg_cost += (vwap_buy - vwap_sell_extra) * (
            amount_usd / ((vwap_buy + vwap_sell_extra) / 2)
        )

    # Base slippage (2-leg spread)
    base_slippage = (vwap_buy - vwap_sell1) * (
        amount_usd / ((vwap_buy + vwap_sell1) / 2)
    )

    # Total slippage cost
    return base_slippage + extra_leg_cost


# --- Load Opportunities ---
filename = r"C:\Users\thoma\PycharmProjects\CryptoDashboard\data\arbitrage\simple_arbitrage_opportunities.jsonl"
base_trade_size = 100_000
taker_fee = 0.001  # 0.1% per leg
number_of_legs = 2

profits_after_fees = []
with open(filename, "r") as f:
    for line in f:
        data = json.loads(line)
        if "last_profit" in data:
            profits_after_fees.append(data["last_profit"])
profits_after_fees = np.array(profits_after_fees)

# --- Back out implied gross spread ---
spread_gross = (profits_after_fees / base_trade_size) + (number_of_legs * taker_fee)


# --- Utility: average observed spread ---
def average_spread(spread_array):
    """Return average observed arbitrage spread in basis points and percent."""
    avg_pct = np.mean(spread_array) * 100  # convert fraction to %
    avg_bps = avg_pct * 100  # 1% = 100 bps
    return avg_pct, avg_bps


avg_pct, avg_bps = average_spread(spread_gross)
print(f"Average observed spread: {avg_pct:.4f}% ({avg_bps:.2f} bps)")

# --- Simulate new trade sizes with slippage ---
trade_sizes = np.linspace(0, 500_000, 50)  # USD amounts to test
total_profits = []
profitable_counts = []

for size in trade_sizes:
    new_profits = []
    for spread in spread_gross:
        gross_profit = size * spread
        fees = size * number_of_legs * taker_fee
        slippage_cost = calc_slippage_cost(asks, bids, size, number_of_legs)
        net_profit = gross_profit - fees - slippage_cost
        new_profits.append(net_profit)
    new_profits = np.array(new_profits)
    total_profits.append(np.sum(new_profits[new_profits > 0]))
    profitable_counts.append(np.sum(new_profits > 0))

# --- Dual-Axis Plot ---
fig, ax1 = plt.subplots(figsize=(10, 6))

color2 = "tab:orange"
ax1.set_ylabel("Number of Profitable Opportunities", color=color2)
ax1.set_xlabel("Trade Size (USD)")
ax1.bar(
    trade_sizes,
    profitable_counts,
    width=(trade_sizes[1] - trade_sizes[0]) * 0.8,
    color=color2,
    alpha=0.4,
    zorder=1,
    label="Profitable Opportunities",
)
ax1.tick_params(axis="y", labelcolor=color2)

# Second axis for line plot
ax2 = ax1.twinx()
color1 = "tab:blue"
ax2.set_xlabel("Trade Size (USD)")
ax2.set_ylabel("Total Profit (USD)", color=color1)
ax2.plot(
    trade_sizes,
    total_profits,
    color=color1,
    linewidth=2,
    label="Total Profit",
    zorder=3,
)
ax2.tick_params(axis="y", labelcolor=color1)

max_idx = np.argmax(total_profits)
max_trade_size = trade_sizes[max_idx]
max_total_profit = total_profits[max_idx]
max_profitable_count = profitable_counts[max_idx]

# Add annotation to the line plot
ax2.scatter(max_trade_size, max_total_profit, color="red", zorder=5)
ax2.annotate(
    f"Max Profit: ${max_total_profit:,.0f}\n"
    f"Trade Size: ${max_trade_size:,.0f}\n"
    f"Profitable Trades: {max_profitable_count}",
    xy=(max_trade_size, max_total_profit),
    xytext=(50, -50),  # offset the text
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="red"),
    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
    fontsize=18,
)

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

# Layout and save high-res
fig.tight_layout()
fig.savefig(
    r"C:\Users\thoma\Documents\Learning\Masters\Year 2\Project\trade_amount_v_opportunities_and_profit.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
