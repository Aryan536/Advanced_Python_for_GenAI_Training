# STEP 0: Imports and data loading
import os
import urllib.request
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid")

FILE = "ecommerce_orders_10k.xlsx"
DATA_URL = "https://raw.githubusercontent.com/anandjha90/Advanced-Python-for-GenAI-Prep-Training/main/data/ecommerce_orders_10k.xlsx"

# 1) use a local copy if it exists (repo cloned locally or file uploaded to Colab)
for candidate in [FILE, os.path.join("..", "data", FILE), os.path.join("data", FILE)]:
    if os.path.exists(candidate):
        FILE = candidate
        break
else:
    # 2) otherwise download it from the GitHub repo
    try:
        urllib.request.urlretrieve(DATA_URL, FILE)
        print("Downloaded dataset from GitHub")
    except Exception as err:
        # 3) last resort: upload it manually in Colab
        print("Download failed:", err)
        from google.colab import files
        files.upload()
print("Using file:", FILE, "| exists:", os.path.exists(FILE))
