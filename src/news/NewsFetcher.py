from newsapi import NewsApiClient
from src.news.SentimentAllocator import SentimentAllocator


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
            self.fetch_latest_news(currency)

    def fetch_latest_news(self, currency):
        self.news_data[currency] = []
        search_query = self.currencies[currency]

        # /v2/top-headlines
        top_headlines = self.client.get_top_headlines(
            q=search_query,
            # sources="bbc-news,the-verge",
            category="business",
            language="en",
            # country="us",
        )

        for headline in top_headlines["articles"]:
            headline["sentiment"] = self.get_sentiment(headline)
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
