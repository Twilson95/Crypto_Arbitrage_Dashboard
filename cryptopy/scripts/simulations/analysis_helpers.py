from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def box_plot_coins(df, group_column):
    mean_profit_order = df.groupby(group_column)["profit"].mean().sort_values()
    df[group_column] = pd.Categorical(
        df[group_column], categories=mean_profit_order.index, ordered=True
    )
    df = df.sort_values(group_column)
    mean_profits = df.groupby(group_column)["profit"].mean()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x=group_column, y="profit", color="lightgray", ax=ax1)
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax1.set_ylim(-100, 100)

    ax1.scatter(
        mean_profits.index,
        mean_profits.values,
        color="blue",
        marker="D",
        label="Mean Profit",
        zorder=5,
    )
    ax2 = ax1.twinx()
    trade_counts = df[group_column].value_counts().reindex(mean_profit_order.index)

    ax2.bar(trade_counts.index, trade_counts.values, color="gray", alpha=0.3)
    ax2.set_ylabel("Number of Trades")

    ax1.set_xticklabels(mean_profit_order.index, rotation=90)
    ax1.set_title("Profit Distribution by Coin with Trade Counts")
    ax1.set_xlabel(group_column.title())
    ax1.set_ylabel("Profit")
    ax1.legend()
    plt.tight_layout()
    plt.show()


def scatter_plot_with_trend(df):
    sns.lmplot(
        data=df,
        x="open_avg_price_ratio",
        y="profit",
        hue="open_direction",
        height=6,
        aspect=1.5,
        scatter_kws={"s": 10},
    )
    plt.title("Scatter Plot of Price Ratio vs Profit with Separate Trend Lines")
    plt.xlabel("Price Ratio")
    plt.ylabel("Profit")
    plt.show()
