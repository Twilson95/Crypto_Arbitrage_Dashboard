import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAllocator:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def generate_sentiment(self, text):
        text = self.preprocess_text(text)
        scores = self.analyzer.polarity_scores(text)
        compound_score = scores["compound"]

        if compound_score >= 0.05:
            sentiment = "positive"
        elif compound_score <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return sentiment

    @staticmethod
    def preprocess_text(text):
        # text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\@\w+|\#", "", text)
        text = re.sub(r"\W", " ", text)
        return text
