from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from datetime import datetime, timedelta
from cryptopy import SentimentAllocator
import os
import json


class NewsFetcher:
    def __init__(self, config):
        self.news_data = {}
        self.api_key = config["NewsApi"]["api_key"]
        self.client = NewsApiClient(self.api_key)
        self.sentiment_allocator = SentimentAllocator()
        # self.fetch_news_sources()
        self.currencies = config["NewsApi"]["pairs"]
        self.caching_folder = "data/news_data/"
        self.fetch_all_latest_news()

    def fetch_news_sources(self):
        # /v2/top-headlines/sources
        sources = self.client.get_sources(
            category="business",
            language="en",
            # country="us",
        )

    def fetch_all_latest_news(self):
        news_api_limit_reached = False
        for currency in self.currencies.keys():
            news = self.get_cached_news(currency)

            # if news api limit reached stop querying it
            if news_api_limit_reached or news is not None:
                continue

            try:
                self.fetch_latest_news(currency)
            except NewsAPIException as e:
                news_api_limit_reached = True
                print(f"{e.args[0].get('message', 'Unknown error')}")
            self.cache_news_data(currency)

    def get_cached_news(self, currency, caching_hrs=1):
        safe_currency = currency.replace("/", "_")
        try:
            file_name = safe_currency + ".json"
            file_path = os.path.join(self.caching_folder, file_name)

            with open(file_path, "r") as f:
                news_data = json.load(f)

            cache_time = datetime.fromisoformat(news_data["cache_time"])

            now = datetime.now()
            time_difference = now - cache_time

            if time_difference < timedelta(hours=caching_hrs):
                # print(f"retrieving cached news data {currency}")
                self.news_data[currency] = news_data["news_items"]
                return news_data["news_items"], False  # Cached data found
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Exception raised: {e}")
            return None

        return None

    def cache_news_data(self, currency):
        news_data = self.news_data.get(currency)

        if news_data:
            safe_currency = currency.replace("/", "_")

            data_to_cache = {
                "cache_time": datetime.now().isoformat(),
                "news_items": news_data,
            }

            file_name = safe_currency + ".json"
            file_path = os.path.join(self.caching_folder, file_name)

            parent_dir = os.path.dirname(file_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)

            # Save to JSON file
            with open(file_path, "w") as f:
                json.dump(data_to_cache, f, indent=4)

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
        news_data = self.news_data.get(currency, {})
        if news_data == {}:
            # find any non-empty news data to return
            for currency, news_data in self.news_data.items():
                if news_data == {}:
                    continue
                else:
                    return news_data
        return news_data

    def add_sentiment_analysis(self):
        currencies = self.news_data.keys()
        for currency in currencies:
            SentimentAllocator.generate_sentiment(self.news_data[currency])
