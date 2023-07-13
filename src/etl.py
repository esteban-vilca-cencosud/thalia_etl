# %%
import os
import numpy as np
import pandas as pd
import threading
import concurrent.futures
import time
from crypto_price import CryptoDataExtractor
from crypto_metrics import CryptoDataTransformation
from indicators import *
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
CHANEL_ID = os.getenv("CHANEL_ID")
BOT_KEY = os.getenv("BOT_KEY")
API_SECRET = os.getenv("API_SECRET")
CRYPTOS_LIST = os.getenv("CRYPTOS_LIST")
TIME = 24 * 200
INTERVAL = "1h"
CRYPTOS = np.load(CRYPTOS_LIST)

PATH = os.getenv("PATH_DATASETS")


# PATH = "./datasets/raw"
# %%
def process(c):
    try:
        extractor = CryptoDataExtractor(save_path=PATH, criptos=[c])
        extractor.from_binance(
            api_key=API_KEY,
            api_secret=API_SECRET,
            time_in_hours=TIME,
            time_interval=INTERVAL,
        )
    except:
        print(f"error: {c}")
        time.sleep(0.01)
        print(f"Call again: {c}")
        process(c)


def check_variation(path, weak=False):
    df = pd.read_csv(path, sep="|")
    lasts = df["lr"].diff()[-3:-1].values
    if lasts[0] < 0 and lasts[1] > 0 and (df["lr"].values[-2] < 0 or weak):
        return 1
    elif lasts[0] > 0 and lasts[1] < 0 and (df["lr"].values[-2] > 0 or weak):
        return -1
    else:
        return 0


def check_adx(path, weak=False):
    df = pd.read_csv(path, sep="|")
    last = df["adx"][-3:-1].values[0]
    mdm = df["mdm"][-3:-1].values[0]
    pdm = df["pdm"][-3:-1].values[0]
    if last >= 13 or weak:
        if mdm > pdm:
            return -1
        elif pdm > mdm:
            return 1
    return 0


import datetime
import datetime
import pytz
import requests


def call_bot(btc_trend, longs, shorts):
    # import telegram

    channel_id = CHANEL_ID
    bot = BOT_KEY
    utc_now = datetime.datetime.now(datetime.timezone.utc)

    # Convert UTC datetime to Brazil timezone (UTC-3)
    brasil_timezone = pytz.timezone("America/Sao_Paulo")
    brasil_now = utc_now.astimezone(brasil_timezone).strftime("%H:%m %d/%m/%Y")

    text = f"Interval: {INTERVAL} - {brasil_now}\nBTC trend: "
    if btc_trend == 0:
        text += "weak trend.\nNot operate."
    elif btc_trend == 1:
        text += "Long weak trend.\nLONGS:\n"
        for e in longs:
            text += str(e) + "\n"
    elif btc_trend == -1:
        text += "Short weak trend.\n"
        for e in shorts:
            text += str(e) + "\n"
    url = f"https://api.telegram.org/{bot}/sendMessage?chat_id={channel_id}&text={text}"

    r = requests.get(url, headers={"Accept": "application/json"})

    print(f"Status Code: {r.status_code}, Content: {r.json()}")


def check_btc():
    if os.path.exists(f"{PATH}/{INTERVAL}/BTCUSDT_silver.csv"):
        adx = check_adx(f"{PATH}/{INTERVAL}/BTCUSDT_silver.csv", weak=True)
        lr = check_variation(f"{PATH}/{INTERVAL}/BTCUSDT_silver.csv", weak=True)
        if lr == -1 and adx == -1:
            return -1
        elif lr == 1 and adx == 1:
            return 1
        else:
            return 0
    return 0


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process, CRYPTOS)

    transformer = CryptoDataTransformation(
        save_path=f"{PATH}/{INTERVAL}", criptos=CRYPTOS
    )
    transformer.readDataset()

    longs = []
    shorts = []
    btc_trend = check_btc()
    for c in CRYPTOS:
        if os.path.exists(f"{PATH}/{INTERVAL}/{c}_silver.csv"):
            adx = check_adx(f"{PATH}/{INTERVAL}/{c}_silver.csv")
            lr = check_variation(f"{PATH}/{INTERVAL}/{c}_silver.csv")
            if lr == 1 and adx == 1 and btc_trend == 1:
                longs.append(c)
            elif lr == -1 and adx == -1 and btc_trend == -1:
                shorts.append(c)
    call_bot(btc_trend, longs, shorts)


# %%
if __name__ == "__main__":
    main()
# %%

# %%
