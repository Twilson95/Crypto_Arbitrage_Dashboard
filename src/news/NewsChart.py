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
            "backgroundColor": "rgba(0, 204, 150, 0.6)",
            "color": "white",
        }
        self.sentiment_neg_style = {
            "if": {"filter_query": "{Sentiment} = 'negative'"},
            "backgroundColor": "rgba(239, 85, 59, 0.6)",
            "color": "white",
        }

    def create_table(self, news_data):
        table_data, description_exists = self.convert_news_data(news_data)
        return self.create_table_layout(table_data, description_exists)

    @staticmethod
    def convert_news_data(news_data):
        table_data = []
        description_exists = False

        for article in news_data:
            description = article["description"]

            if description:
                description_exists = True

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
        return table_data, description_exists

    def create_table_layout(self, table_data, description_exists):
        columns = [
            {"name": "Source", "id": "Source"},
            # {"name": "Author", "id": "Author"},
            {"name": "Title", "id": "Title"},
            # {"name": "Description", "id": "Description"},
            {"name": "URL", "id": "URL", "presentation": "markdown"},
            {"name": "Published", "id": "Published"},
        ]

        # if description_exists:
        #     columns.insert(2, {"name": "Description", "id": "Description"})

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
                {
                    "selector": ".dash-table-container .dash-spreadsheet-container tbody tr:hover",
                    "rule": "background-color: inherit !important;",  # Disable hover highlighting
                },
            ],
            style_table=style_table,
            style_cell=style_cell,
            style_header={"backgroundColor": "rgb(30, 30, 30)", "color": "white"},
            style_data={"backgroundColor": "rgb(50, 50, 50)", "color": "white"},
            style_data_conditional=[
                self.url_style,
                self.sentiment_pos_style,
                self.sentiment_neg_style,
                {"if": {"column_id": "Source"}, "width": "15%", "whiteSpace": "normal"},
                {"if": {"column_id": "Title"}, "width": "50%"},
            ],
        )
