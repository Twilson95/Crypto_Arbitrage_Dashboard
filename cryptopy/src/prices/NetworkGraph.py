import plotly.graph_objects as go
import networkx as nx
import numpy as np
from cryptopy.src.prices.helper_functions import format_amount


def create_network_graph(live_prices, arbitrage_opportunities=None):
    """
    Creates a Plotly network graph from a dictionary of live prices with labeled edges,
    with USD in the center and other coins arranged in a circle around it. Highlights
    triangular arbitrage routes if provided.

    Parameters:
        live_prices (dict): A dictionary where keys are currency pairs and values are exchange rates.
                            Example: {'ETH/BTC': 0.065, 'BTC/USD': 40000}
        arbitrage_opportunities (list): Optional. A list of dictionaries, each containing an arbitrage opportunity,
                                        including the 'path' and 'summary_header' with 'total_profit'.

    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure object for the network graph.
    """

    # Initialize a directed graph
    G = create_graph_from_prices(live_prices)

    # Generate positions and edge data
    pos = generate_node_positions(G)
    edge_x, edge_y, edge_text = generate_edge_positions(G, pos)

    # Initialize a list to collect all traces
    traces = []

    # Handle arbitrage opportunities if provided
    if arbitrage_opportunities:
        for arbitrage in arbitrage_opportunities:
            if arbitrage is None:
                continue

            arbitrage_route = arbitrage.get("path", None)
            total_profit = arbitrage.get("summary_header", {}).get("total_profit", None)

            if arbitrage_route:
                arb_edge_x, arb_edge_y, arb_edge_text = (
                    generate_arbitrage_edge_positions(G, pos, arbitrage_route)
                )

                # Create Plotly traces for arbitrage route
                arb_edge_trace = create_arbitrage_edge_trace(
                    arb_edge_x, arb_edge_y, total_profit
                )
                arb_edge_labels = create_arbitrage_edge_labels(
                    arb_edge_x, arb_edge_y, arb_edge_text
                )

                last_node = arbitrage_route[-1][1]

                # Add profit annotation near the last node in the arbitrage route
                # profit_annotation = go.Scatter(
                #     x=[pos[last_node][0]],  # Access single node
                #     y=[pos[last_node][1]],  # Access single node
                #     text=[
                #         (
                #             f"Profit: ${total_profit:.2f}"
                #             if total_profit is not None
                #             else ""
                #         )
                #     ],
                #     mode="text",
                #     showlegend=False,
                # )

                # Append the arbitrage traces
                traces.append(arb_edge_trace)
                traces.append(arb_edge_labels)
                # traces.append(profit_annotation)

    base_traces = [
        create_edge_trace(edge_x, edge_y),
        # create_edge_labels(G, pos, edge_text),
        create_node_trace(G, pos),
    ]

    traces += base_traces

    # Customize and create the figure with all traces
    fig = create_figure(traces)

    return fig


def create_graph_from_prices(live_prices):
    G = nx.DiGraph()
    for pair, rate in live_prices.items():
        base, quote = pair.split("/")
        G.add_edge(base, quote, weight=rate)
    return G


def generate_node_positions(G):
    nodes = list(G.nodes)
    center_node = "USD"

    if center_node not in nodes:
        nodes.append(center_node)

    pos = {center_node: (0, 0)}  # Center USD

    surrounding_nodes = [node for node in nodes if node != center_node]
    num_surrounding_nodes = len(surrounding_nodes)

    angle_gap = (
        2 * np.pi / num_surrounding_nodes
    )  # Correct angle gap for surrounding nodes
    radius = 1

    for i, node in enumerate(surrounding_nodes):
        angle = i * angle_gap
        pos[node] = (radius * np.cos(angle), radius * np.sin(angle))

    return pos


def generate_edge_positions(G, pos):
    edge_x = []
    edge_y = []
    edge_text = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{format_amount(edge[2]['weight'])}")

    return edge_x, edge_y, edge_text


def generate_arbitrage_edge_positions(G, pos, arbitrage_route, shorten_factor=0.05):
    arb_edge_x = []
    arb_edge_y = []
    arb_edge_text = []

    if not arbitrage_route:
        return arb_edge_x, arb_edge_y, arb_edge_text

    for edge in arbitrage_route:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Calculate direction vector
        dx = x1 - x0
        dy = y1 - y0

        # Shorten the end point by the shorten_factor
        x1_short = x0 + (1 - shorten_factor) * dx
        y1_short = y0 + (1 - shorten_factor) * dy

        arb_edge_x.extend([x0, x1_short, None])
        arb_edge_y.extend([y0, y1_short, None])

        if G.has_edge(edge[0], edge[1]):
            rate = G[edge[0]][edge[1]]["weight"]
            arb_edge_text.append(f"{format_amount(rate)}")
        elif G.has_edge(edge[1], edge[0]):
            rate = G[edge[1]][edge[0]]["weight"]
            arb_edge_text.append(f"{format_amount(1/rate)}")
        else:
            arb_edge_text.append("")

    return arb_edge_x, arb_edge_y, arb_edge_text


def create_edge_trace(edge_x, edge_y):
    return go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )


def create_arbitrage_edge_trace(arb_edge_x, arb_edge_y, total_profit):
    color = "green" if total_profit > 0 else "orange"

    return go.Scatter(
        x=arb_edge_x,
        y=arb_edge_y,
        line=dict(width=2, color=color),
        hoverinfo="none",
        mode="lines+markers",
        marker=dict(
            size=10,
            color=color,
            symbol="arrow-bar-up",  # Use an arrow symbol to show direction
            angleref="previous",
        ),
    )


def create_arbitrage_edge_labels(
    arb_edge_x, arb_edge_y, arb_edge_text, offset=0.05, min_distance=0.1
):
    edge_label_x = []
    edge_label_y = []
    last_label_position = None

    for i in range(0, len(arb_edge_x) - 2, 3):
        x0, y0 = arb_edge_x[i], arb_edge_y[i]
        x1, y1 = arb_edge_x[i + 1], arb_edge_y[i + 1]

        # Calculate label position with the default offset
        offset_x, offset_y = compute_perpendicular_offset(x0, y0, x1, y1, offset)

        # Check the distance from the last label position
        if last_label_position:
            last_x, last_y = last_label_position
            distance = np.sqrt((offset_x - last_x) ** 2 + (offset_y - last_y) ** 2)
            if distance < min_distance:
                # If too close, apply offset in the opposite direction
                offset_x, offset_y = compute_perpendicular_offset(
                    x0, y0, x1, y1, -offset
                )

        # Update last label position
        last_label_position = (offset_x, offset_y)

        edge_label_x.append(offset_x)
        edge_label_y.append(offset_y)

    # Create a Plotly scatter plot for edge labels
    return go.Scatter(
        x=edge_label_x,
        y=edge_label_y,
        text=arb_edge_text,
        mode="text",
        textposition="middle center",
        hoverinfo="text",
    )


def create_edge_labels(G, pos, edge_text):
    edge_label_x = [(pos[edge[0]][0] + pos[edge[1]][0]) / 2 for edge in G.edges()]
    edge_label_y = [(pos[edge[0]][1] + pos[edge[1]][1]) / 2 for edge in G.edges()]

    return go.Scatter(
        x=edge_label_x,
        y=edge_label_y,
        text=edge_text,
        mode="text",
        textposition="middle center",
        hoverinfo="text",
    )


def create_node_trace(G, pos):
    node_x = []
    node_y = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    return go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        marker=dict(
            colorscale="YlGnBu",
            size=20,
            line_width=2,
        ),
    )


def create_figure(traces):
    return go.Figure(
        data=traces,  # Changed from [traces] to traces directly
        layout=go.Layout(
            title="Cryptocurrency Exchange Network",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            template="plotly_dark",
            annotations=[
                dict(
                    text="Live exchange rates between cryptocurrencies",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )


def compute_perpendicular_offset(x0, y0, x1, y1, offset=0.05):
    """
    Computes a perpendicular offset for a point on a line.
    """
    # Calculate the direction vector of the line
    dx = x1 - x0
    dy = y1 - y0

    # Normalize the direction vector
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:  # Prevent division by zero
        return 0, 0

    dx /= length
    dy /= length

    # Compute the perpendicular vector (rotate 90 degrees)
    perp_dx = -dy
    perp_dy = dx

    # Apply the offset
    offset_x = x0 + dx * 0.5 + perp_dx * offset
    offset_y = y0 + dy * 0.5 + perp_dy * offset

    return offset_x, offset_y
