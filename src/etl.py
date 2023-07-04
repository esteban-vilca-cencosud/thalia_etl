#%%
import os
import numpy as np
import pandas as pd
import threading
import concurrent.futures
import time
from crypto_price import CryptoDataExtractor
from crypto_metrics import CryptoDataTransformation

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
CHANEL_ID = os.getenv("CHANEL_ID")
BOT_KEY = os.getenv("BOT_KEY")
API_SECRET = os.getenv("API_SECRET")
CRYPTOS_LIST = os.getenv("CRYPTOS_LIST")
TIME = 24*200
INTERVAL = "1h"
CRYPTOS = np.load(CRYPTOS_LIST)

PATH = os.getenv("PATH_DATASETS")
# PATH = "./datasets/raw"
#%%
def process(c):
    try:
        extractor = CryptoDataExtractor(save_path=PATH,criptos=[c])
        extractor.from_binance(api_key=API_KEY,api_secret=API_SECRET,
                                time_in_hours=TIME, time_interval=INTERVAL)
    except:
        print(f"error: {c}")
        time.sleep(0.01)
        print(f"Call again: {c}")
        process(c)
def check_variation(path):
    df = pd.read_csv(path,sep="|")
    lasts=df["lr"].diff()[-2:].values
    if lasts[0]<0 and lasts[1]>0:
        return 1
    elif lasts[0]>0 and lasts[1]<0:
        return -1
    else:
        return 0
def check_adx(path):
    df = pd.read_csv(path,sep="|")
    last=df["adx"][-1:].values[0]
    mdm=df["mdm"][-1:].values[0]
    pdm=df["pdm"][-1:].values[0]
    if last >13:
        if mdm>pdm:
            return -1
        elif pdm>mdm:
            return 1
    return 0
import datetime
import datetime
import pytz

def call_bot(text):
    # import telegram
    import requests

    channel_id = CHANEL_ID
    bot = BOT_KEY
    # # Send a text message
    # bot.send_message(chat_id=channel_id, text='Hello, World!')

    # # Send a photo
    # bot.send_photo(chat_id=channel_id, photo=open('/path/to/photo.jpg', 'rb'))

    # # Send a vide
    # bot.send_video(chat_id=channel_id, video=open('/path/to/video.mp4', 'rb'))

    # # Send a document
    # bot.send_document(chat_id=channel_id, document=open('/path/to/document.pdf', 'rb'))


    # Get the current UTC datetime
    utc_now = datetime.datetime.now(datetime.timezone.utc)

    # Convert UTC datetime to Brazil timezone (UTC-3)
    brasil_timezone = pytz.timezone("America/Sao_Paulo")
    brasil_now = utc_now.astimezone(brasil_timezone)

    text = brasil_now.strftime("%H:%m %d/%m/%Y")+"\n"+ text
    url = f"https://api.telegram.org/{bot}/sendMessage?chat_id={channel_id}&text={text}"

    r = requests.get(url, 
                    headers={'Accept': 'application/json'})

    print(f"Status Code: {r.status_code}, Content: {r.json()}")

def main():

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process, CRYPTOS)

    transformer = CryptoDataTransformation(save_path=f"{PATH}/{INTERVAL}",criptos=CRYPTOS)
    transformer.readDataset()

    longs = []
    shorts = []
    for c in CRYPTOS:
        if os.path.exists(f"{PATH}/{INTERVAL}/{c}_silver.csv"):
            adx = check_adx(f"{PATH}/{INTERVAL}/{c}_silver.csv")
            lr = check_variation(f"{PATH}/{INTERVAL}/{c}_silver.csv")
            if(lr==-1 and adx==-1):
                print("LONG: ",c)
                longs.append(c)
            elif(lr==1 and adx==1):
                print("SHORT:",c)
                shorts.append(c)
    call_bot("LONGS:\n"+str(longs))
    call_bot("SHORTS:\n"+str(shorts))
if __name__=="__main__":
    main()
# %%

# %%

