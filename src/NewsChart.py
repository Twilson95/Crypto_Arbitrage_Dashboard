import requests
import plotly.graph_objs as go
from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc


class NewsChart:
    def __init__(self):
        self.data = 0

    def create_table(self, news_data):
        # print("news", news_data)
        table_data = self.convert_news_data(news_data)
        # print("table_data", table_data)
        return self.create_table_layout(table_data)

    @staticmethod
    def convert_news_data(news_data):
        table_data = []
        for article in news_data:
            table_data.append(
                {
                    "Source": article["source"]["name"],
                    "Author": article["author"],
                    "Title": article["title"],
                    "Description": article["description"],
                    "URL": article["url"],
                    "Published At": article["publishedAt"],
                }
            )
        return table_data

    @staticmethod
    def create_table_layout(table_data):
        print(table_data)
        columns = [
            {"name": "Source", "id": "Source"},
            {"name": "Author", "id": "Author"},
            {"name": "Title", "id": "Title"},
            {"name": "Description", "id": "Description"},
            {"name": "URL", "id": "URL"},
            {"name": "Published At", "id": "Published At"},
        ]

        return dash_table.DataTable(
            id="news-table",
            columns=columns,
            data=table_data,
            style_table={
                # "maxHeight": "400px",  # Set a maximum height for the table container
                "overflowY": "scroll",  # Enable vertical scrolling
                "overflowX": "auto",  # Enable horizontal scrolling if needed
                "display": "inline-block",
                "width": "100%",
            },
            style_cell={
                # "height": "auto",
                "minWidth": "100px",
                "width": "100px",
                "maxWidth": "100%",
                "whiteSpace": "normal",
            },
            style_header={"backgroundColor": "rgb(30, 30, 30)", "color": "white"},
            style_data={"backgroundColor": "rgb(50, 50, 50)", "color": "white"},
            style_data_conditional=[
                {
                    "if": {"column_id": "URL"},
                    "textDecoration": "underline",
                    "color": "#0074D9",
                }
            ],
        )
