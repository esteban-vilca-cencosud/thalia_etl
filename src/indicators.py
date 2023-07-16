import numpy as np
import pandas as pd
from typing import List
from abc import ABC, abstractmethod
class Indicator():
    tendency=0
    long=0
    short=0
    def __init__(self, tendency=0, long=0, short=0) -> None:
        self.tendency = tendency
        self.long = long
        self.short = short

def get_last(df,n=3):
    return df.iloc[-n:]

def adx_limit(df):
    l = 13
    r = all(df["adx"].iloc[-5:] > l)
    return Indicator(tendency=int(r))
    
def lr_swap(df:pd.DataFrame):
    df2 = df[["lr"]].diff()
    df2 = get_last(df2)
    df = get_last(df)
    result = df2[["lr"]].prod()
class PipelineIndicators:
    indicators = {}

    def __init__(self) -> None:
        self.add_indicator("adx_limit","adx_thresh",adx_limit)
        # df = pd.read_csv(path, sep="|")
        # last = df["adx"][-3:-1].values[0]
        # mdm = df["mdm"][-3:-1].values[0]
        # pdm = df["pdm"][-3:-1].values[0]
        # if last >= 13 or weak:
        #     if mdm > pdm:
        #         return -1
        #     elif pdm > mdm:
        #         return 1
        # return 0

    def add_indicator(self, name, description, f):
        self.indicators[name] = {"execute": f, "description": description}

    def execute(self, pipeline, dataset):
        new_data = dataset
        results = []
        for indicator in pipeline:
            results.append(indicator["execute"](new_data))

