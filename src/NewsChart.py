import requests
import plotly.graph_objs as go


class NewsChart:
    def __init__(self):
        self.news_data = {}

    def get_news(self, currency):
        if currency not in self.news_data:
            url = f"https://newsapi.org/v2/everything?q={currency}&apiKey=YOUR_NEWS_API_KEY"
            response = requests.get(url)
            self.news_data[currency] = response.json()
        return self.news_data[currency]

    @staticmethod
    def create_chart(news):
        fig = go.Figure()
        for article in news["articles"]:
            fig.add_trace(
                go.Scatter(
                    x=[article["publishedAt"]],
                    y=[0],
                    text=article["title"],
                    mode="markers+text",
                    textposition="top center",
                )
            )
        return fig
