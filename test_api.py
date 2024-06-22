import requests
import bitmex
import json
import time
import hashlib
import hmac
import urllib.parse

bitmex_api_key = 'Hi7WUgzxyzCRY_1BJ0e7meab'
bitmex_api_secret = 'A64KwkbRqURgFAmarfF758ceAVtFoIJWZIXe5lpRIeld1FxD'

client = bitmex.bitmex(test=False, api_key= bitmex_api_key, api_secret=bitmex_api_secret)
# client = bitmex(test=False, api_key=API_KEY, api_secret=API_SECRET)


positions = client.Instrument.Instrument_get(filter=json.dumps({"symbol": 'XBTUSD'})).result()[0][0]
bitmex_btc = {}
print(positions['lastPrice'])
#
# bitmex_btc["markPrice"] = positions["markPrice"]
#
# print(positions)

# Generates an API signature.
# A signature is HMAC_SHA256(secret, verb + path + expires + data), hex encoded.
# Verb must be uppercased, url is relative, expires must be unix timestamp (in seconds)
# and the data, if present, must be JSON without whitespace between keys.
def generate_signature(secret, verb, url, expires, data):
    """Generate a request signature compatible with BitMEX."""
    # Parse the url so we can remove the base and extract just the path.
    parsedURL = urllib.parse.urlparse(url)
    path = parsedURL.path
    if parsedURL.query:
        path = path + '?' + parsedURL.query

    if isinstance(data, (bytes, bytearray)):
        data = data.decode('utf8')

    message = bytes(verb + path + str(expires) + data, 'utf-8')
    print("Computing HMAC: %s" % message)

    signature = hmac.new(bytes(secret, 'utf-8'), message, digestmod=hashlib.sha256).hexdigest()
    return signature

#
# Testing
#
# expires = 1518064236
# Or you might generate it like so:
expires = int(round(time.time()) + 5)

# Prints 'c7682d435d0cfe87c16098df34ef2eb5a549d4c5a3c2b1f0f77b8af73423bf00'
# print(generate_signature(bitmex_api_secret, 'GET', '/api/v1/XBTUSD', expires, ''))