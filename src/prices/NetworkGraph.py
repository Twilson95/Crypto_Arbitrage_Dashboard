import plotly.graph_objects as go
import networkx as nx
import numpy as np
from src.prices.helper_functions import format_amount


def create_network_graph(live_prices, arbitrage_route=None):
    """
    Creates a Plotly network graph from a dictionary of live prices with labeled edges,
    with USD in the center and other coins arranged in a circle around it. Highlights
    a triangular arbitrage route if provided.

    Parameters:
        live_prices (dict): A dictionary where keys are currency pairs and values are exchange rates.
                            Example: {'ETH/BTC': 0.065, 'BTC/USD': 40000}
        arbitrage_route (list): Optional. A list of tuples representing the arbitrage route edges.
                                Example: [('BTC', 'USD'), ('USD', 'ETH'), ('ETH', 'BTC')]

    Returns:
        fig (plotly.graph_objects.Figure): Plotly figure object for the network graph.
    """
    # Initialize a directed graph
    G = create_graph_from_prices(live_prices)

    # Generate positions and edge data
    pos = generate_node_positions(G)
    edge_x, edge_y, edge_text = generate_edge_positions(G, pos)
    arb_edge_x, arb_edge_y, arb_edge_text = generate_arbitrage_edge_positions(
        G, pos, arbitrage_route
    )

    # Create Plotly traces
    edge_trace = create_edge_trace(edge_x, edge_y)
    arb_edge_trace = create_arbitrage_edge_trace(arb_edge_x, arb_edge_y)
    edge_labels = create_edge_labels(G, pos, edge_text)
    node_trace = create_node_trace(G, pos)

    # Customize and create the figure
    fig = create_figure(edge_trace, arb_edge_trace, edge_labels, node_trace)

    return fig


def create_graph_from_prices(live_prices):
    G = nx.DiGraph()
    for pair, price in live_prices.items():
        rate = price.close[-1]
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


def generate_arbitrage_edge_positions(G, pos, arbitrage_route, shorten_factor=0.1):
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
            arb_edge_text.append(f"{G[edge[0]][edge[1]]['weight']:.4f}")
        else:
            arb_edge_text.append("")

    return arb_edge_x, arb_edge_y, arb_edge_text


def create_edge_trace(edge_x, edge_y):

    return go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )


def create_arbitrage_edge_trace(arb_edge_x, arb_edge_y):

    return go.Scatter(
        x=arb_edge_x,
        y=arb_edge_y,
        line=dict(width=2, color="green"),
        hoverinfo="none",
        mode="lines+markers",
        marker=dict(
            size=10,
            color="green",
            symbol="arrow-bar-up",
            angleref="previous",  # Use an arrow symbol to show direction
        ),
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


def create_figure(edge_trace, arb_edge_trace, edge_labels, node_trace):

    return go.Figure(
        data=[edge_trace, arb_edge_trace, edge_labels, node_trace],
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
