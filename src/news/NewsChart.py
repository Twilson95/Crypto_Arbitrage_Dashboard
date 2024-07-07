from dash import dash_table
from src.layout.layout_styles import style_table, style_cell


class NewsChart:
    def __init__(self):
        self.url_style = {
            "if": {"column_id": "URL"},
            "textDecoration": "underline",
            "color": "#0074D9",
        }
        self.sentiment_pos_style = {
            "if": {"filter_query": "{Sentiment} = 'positive'"},
            "backgroundColor": "#d4edda",
            "color": "white",
        }
        self.sentiment_neg_style = {
            "if": {"filter_query": "{Sentiment} = 'negative'"},
            "backgroundColor": "#f8d7da",
            "color": "white",
        }

    def create_table(self, news_data):
        table_data = self.convert_news_data(news_data)
        return self.create_table_layout(table_data)

    @staticmethod
    def convert_news_data(news_data):
        table_data = []
        for article in news_data:
            table_data.append(
                {
                    "Source": article["source"]["name"],
                    # "Author": article["author"],
                    "Title": article["title"],
                    "Description": article["description"],
                    "URL": f"[Click Here]({article['url']})",
                    "Published": article["publishedAt"][:10],
                    "Sentiment": article["sentiment"],
                }
            )
        return table_data

    def create_table_layout(self, table_data):
        columns = [
            {"name": "Source", "id": "Source"},
            # {"name": "Author", "id": "Author"},
            {"name": "Title", "id": "Title"},
            {"name": "Description", "id": "Description"},
            {"name": "URL", "id": "URL", "presentation": "markdown"},
            {"name": "Published", "id": "Published"},
        ]

        return dash_table.DataTable(
            id="news-table",
            columns=columns,
            data=table_data,
            fixed_rows={"headers": True},
            css=[
                {
                    "selector": ".dash-spreadsheet-container",
                    "rule": "overflow: hidden !important;",  # Force hide overflow
                },
                #     {
                #         "selector": ".dash-table-container .dash-spreadsheet-container",
                #         "rule": "line-height: unset !important;",  # Override line-height
                #     },
            ],
            style_table=style_table,
            style_cell=style_cell,
            style_header={"backgroundColor": "rgb(30, 30, 30)", "color": "white"},
            style_data={"backgroundColor": "rgb(50, 50, 50)", "color": "white"},
            style_data_conditional=[
                self.url_style,
                self.sentiment_pos_style,
                self.sentiment_neg_style,
            ],
        )
