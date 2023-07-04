#%%
import os
from crypto_price import CryptoDataExtractor
from crypto_metrics import CryptoDataTransformation

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
TIME = 24*1200
INTERVAL = "4h"
CRYPTOS = ["BTCUSDT","ETHUSDT","MATICUSDT","TRXUSDT","XRPUSDT"]
PATH = "../datasets/raw"
extractor = CryptoDataExtractor(save_path=PATH,criptos=CRYPTOS)
extractor.from_binance(api_key=API_KEY,api_secret=API_SECRET,
                        time_in_hours=TIME, time_interval=INTERVAL)
transformer = CryptoDataTransformation(save_path=f"{PATH}/{INTERVAL}",criptos=CRYPTOS)
transformer.readDataset()
# %%
