from newsapi import NewsApiClient
from datetime import datetime
from src.SentimentAllocator import SentimentAllocator


class NewsFetcher:
    def __init__(self, config):
        self.news_data = {}
        self.api_key = config["NewsApi"]["api_key"]
        self.client = NewsApiClient(self.api_key)
        self.sentiment_allocator = SentimentAllocator()
        self.fetch_news_sources()
        # self.fetch_all_latest_news()
        self.fetch_latest_news("bitcoin")
        # self.add_sentiment_analysis()

    def fetch_news_sources(self):
        # /v2/top-headlines/sources
        sources = self.client.get_sources(
            category="business",
            language="en",
            # country="us",
        )
        # print(sources)

    def fetch_all_latest_news(self):
        currencies = self.news_data.keys()
        for currency in currencies:
            self.fetch_latest_news(currency)

    def fetch_latest_news(self, currency):
        currency = "bitcoin"
        self.news_data[currency] = []

        # /v2/top-headlines
        top_headlines = self.client.get_top_headlines(
            q=currency,
            # sources="bbc-news,the-verge",
            category="business",
            language="en",
            # country="us",
        )
        # print("top_headlines", top_headlines[0])
        for headline in top_headlines["articles"]:
            self.news_data[currency].append(headline)

        # /v2/everything
        # all_articles = self.client.get_everything(
        #     q="bitcoin",
        #     sources="bbc-news,the-verge",
        #     domains="bbc.co.uk,techcrunch.com",
        #     from_param="2017-12-01",
        #     to=datetime.today().strftime("yyyy-MM-dd"),
        #     language="en",
        #     sort_by="relevancy",
        #     page=2,
        # )
        # print("all_articles", all_articles)

    def get_news_data(self, currency):
        return self.news_data[currency]

    def add_sentiment_analysis(self):
        currencies = self.news_data.keys()
        for currency in currencies:
            SentimentAllocator.generate_sentiment(self.news_data[currency])
