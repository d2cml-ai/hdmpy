
import pandas as pd
import os

path = os.path.dirname(__file__)

def load_cps2012():

    return pd.read_csv(os.path.join(path, "data/cps2012.csv"))

def load_AJR():

    return pd.read_csv(os.path.join(path, "data/AJR.csv"))

