from cryptopy.src.prices.OHLCData import OHLCData
from datetime import datetime
from dateutil.tz import tzutc

news = [
    {
        "Source": "Ambcrypto.com",
        # "Author": "Ishika Kumari",
        "Title": "Bitcoin poised for growth as Japanese Yen hits record low - AMBCrypto News",
        "Description": "Will Japan's weak Yen boost Bitcoin's potential or is it just mere speculations?",
        "URL": "[Click Here](https://ambcrypto.com/bitcoin-poised-for-growth-as-japanese-yen-hits-record-low/)",
        "Published": "2024-06-28",
    },
    {
        "Source": "Ambcrypto.com",
        # "Author": "Ishika Kumari",
        "Title": "Bitcoin poised for growth as Japanese Yen hits record low - AMBCrypto News",
        "Description": "Will Japan's weak Yen boost Bitcoin's potential or is it just mere speculations?",
        "URL": "[Click Here](https://ambcrypto.com/bitcoin-poised-for-growth-as-japanese-yen-hits-record-low/)",
        "Published": "2024-06-28",
    },
    {
        "Source": "Ambcrypto.com",
        # "Author": "Ishika Kumari",
        "Title": "Bitcoin poised for growth as Japanese Yen hits record low - AMBCrypto News",
        "Description": "Will Japan's weak Yen boost Bitcoin's potential or is it just mere speculations?",
        "URL": "[Click Here](https://ambcrypto.com/bitcoin-poised-for-growth-as-japanese-yen-hits-record-low/)",
        "Published": "2024-06-28",
    },
    {
        "Source": "Ambcrypto.com",
        # "Author": "Ishika Kumari",
        "Title": "Bitcoin poised for growth as Japanese Yen hits record low - AMBCrypto News",
        "Description": "Will Japan's weak Yen boost Bitcoin's potential or is it just mere speculations?",
        "URL": "[Click Here](https://ambcrypto.com/bitcoin-poised-for-growth-as-japanese-yen-hits-record-low/)",
        "Published": "2024-06-28",
    },
]

live_data = OHLCData()
live_data.datetime = [
    datetime(2024, 6, 30, 8, 12, 8, 474618, tzinfo=tzutc()),
]
live_data.open = [61293.0]
live_data.high = [61293.0]
live_data.low = [61293.0]
live_data.close = [61293.0]

historical_data = OHLCData()
historical_data.datetime = [
    datetime(2024, 6, 30, 8, 12, 8, 474618, tzinfo=tzutc()),
]
historical_data.open = [61293.0]
historical_data.high = [61293.0]
historical_data.low = [61293.0]
historical_data.close = [61293.0]
