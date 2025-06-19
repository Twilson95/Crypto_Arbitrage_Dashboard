from setuptools import setup, find_packages

setup(
    name="Crypto_Dashboard",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "dash~=3.0.4",
        "pandas~=2.2.2",
        "vaderSentiment~=3.3.2",
        "plotly~=5.22.0",
        "ccxt~=4.4.14",
        "pytz~=2024.1",
        "networkx~=3.3",
        "numpy~=1.26.4",
        "statsmodels~=0.14.2",
        "PyYAML~=6.0.1",
        "setuptools~=70.3.0",
        "krakenex~=2.2.2",
    ],
    test_suite="tests",  # This points to the folder where your test cases are located
)
