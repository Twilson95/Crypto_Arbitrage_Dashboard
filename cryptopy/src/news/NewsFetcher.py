from newsapi import NewsApiClient
from datetime import datetime, timedelta
from cryptopy import SentimentAllocator
import pandas as pd


class NewsFetcher:
    def __init__(self, config):
        self.news_data = {}
        self.api_key = config["NewsApi"]["api_key"]
        self.client = NewsApiClient(self.api_key)
        self.sentiment_allocator = SentimentAllocator()
        # self.fetch_news_sources()
        self.currencies = config["NewsApi"]["pairs"]
        self.fetch_all_latest_news()

    def fetch_news_sources(self):
        # /v2/top-headlines/sources
        sources = self.client.get_sources(
            category="business",
            language="en",
            # country="us",
        )

    def fetch_all_latest_news(self):
        for currency in self.currencies.keys():
            news = self.get_cached_news(currency)
            if news is None:
                self.fetch_latest_news(currency)

    def get_cached_news(self, currency):
        try:
            folder_path = r"../../data/news_data/"
            file_name = currency + ".csv"
            news_df = pd.read_csv(folder_path + file_name)
        except:
            news_df = None
        return news_df

    def fetch_latest_news(self, currency):
        self.news_data[currency] = []
        search_query = self.currencies[currency]

        today = datetime.utcnow().date()
        days_back = 2
        start_date = today - timedelta(days=days_back)

        everything = self.client.get_everything(
            q=search_query,
            from_param=start_date.isoformat(),
            to=today.isoformat(),
            language="en",
            sort_by="publishedAt",
            page_size=10,
        )

        for article in everything["articles"]:
            article["sentiment"] = self.get_sentiment(article)
            self.news_data[currency].append(article)

    def get_sentiment(self, headline):
        text = headline.get("description") or headline.get("title")

        if text is None:
            return None

        sentiment = self.sentiment_allocator.generate_sentiment(text)
        return sentiment

    def get_news_data(self, currency):
        return self.news_data[currency]

    def add_sentiment_analysis(self):
        currencies = self.news_data.keys()
        for currency in currencies:
            SentimentAllocator.generate_sentiment(self.news_data[currency])
