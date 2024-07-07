import streamlit as st
import configparser
import pandas as pd
from src.FilterComponents import FilterComponent
from src.TechnicalIndicators import TechnicalIndicators
from src.PriceChart import PriceChart
from src.NewsFetcher import NewsFetcher
from src.NewsChart import NewsChart
from src.DataManager import DataManager

st.set_page_config(page_title="Crypto Dashboard", layout="wide")

config = configparser.ConfigParser()
config.read("./src/Config.ini")

filter_component = FilterComponent()
technical_indicators = TechnicalIndicators()
data_manager = DataManager(config)
news_fetcher = NewsFetcher(config)
price_chart = PriceChart()
news_chart = NewsChart()

st.title("Crypto Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
exchange = st.sidebar.selectbox(
    "Select an exchange", filter_component.get_exchange_names(), index=0
)
currency = st.sidebar.selectbox(
    "Select a currency", filter_component.get_currency_names(), index=0
)
indicators = st.sidebar.multiselect(
    "Select Technical Indicators", technical_indicators.get_indicator_options()
)

# Update data
data_manager.fetch_all_live_prices()

# Historic Price Chart
if currency and exchange:
    prices = data_manager.get_historical_prices(exchange, currency)
    if prices:
        indicators_data = technical_indicators.apply_indicators(prices, indicators)
        historic_chart = price_chart.create_chart(
            prices, indicators_data, title="Historic Price", mark_limit=60
        )
        st.plotly_chart(historic_chart)

# Live Price Chart
if currency and exchange:
    live_prices = data_manager.get_live_prices(exchange, currency)
    if live_prices:
        live_chart = price_chart.create_chart(
            live_prices, mark_limit=20, title="Live Price"
        )
        st.plotly_chart(live_chart)

# News Table
if currency:
    news = news_fetcher.get_news_data(currency)
    if news:
        news_table = news_chart.create_table(news)
        st.dataframe(pd.DataFrame(news_table))

# Interval for live data update
# In Streamlit, you can't directly use intervals like in Dash, but you can set up a refresh mechanism or a button to refresh the data.
st.sidebar.button("Refresh Data", on_click=lambda: st.experimental_rerun())

# Custom CSS for scrollbars (Streamlit automatically supports custom CSS)
custom_scrollbar_css = """
<style>
/* Custom scrollbars for WebKit browsers (Chrome, Safari) */
::-webkit-scrollbar {
    width: 6px; /* Width of the entire scrollbar */
    height: 6px; /* Height of the entire scrollbar */
}

::-webkit-scrollbar-track {
    background: transparent; /* Set the scrollbar track background to transparent */
}

::-webkit-scrollbar-thumb {
    background-color: #4974a5; /* Color of the scrollbar handle */
    border-radius: 1px; /* Roundness of the scrollbar handle */
    border: 0px solid transparent; /* Padding around scrollbar handle */
}

::-webkit-scrollbar-thumb:active {
    background: #555; /* Color of the scrollbar handle on click */
}

/* Custom scrollbars for Firefox */
* {
    scrollbar-width: thin; /* Width of the scrollbar */
    scrollbar-color: #4974a5 transparent; /* Color of the scrollbar handle and transparent track */
}
</style>
"""
st.markdown(custom_scrollbar_css, unsafe_allow_html=True)
