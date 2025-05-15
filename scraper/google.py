from pytrends.request import TrendReq
import pandas as pd
import time
import random

# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=360)
keywords = [  "linen shirt" , "skirt"]

proxies = {
    'http': 'http://192.168.1.100:8080',
    'https': 'http://192.168.1.100:8080'
}

data_frames = []  # <- you forgot this earlier

for kw in keywords:
    try:
        pytrends.build_payload([kw], timeframe='today 3-m')
        df = pytrends.interest_over_time()
        if not df.empty:
            df['keyword'] = kw
            data_frames.append(df)
            print(f"[✔] Collected: {kw}")
        else:
            print(f"[⚠] Empty data: {kw}")
        time.sleep(random.randint(30, 60))  # prevent rate limiting
    except Exception as e:
        print(f"[✘] Failed: {kw} -> {e}")

# Combine all data
if data_frames:
    result_df = pd.concat(data_frames)
    result_df.to_csv("trending_keywords.csv")
    print("[📁] Saved to trending_keywords.csv")
else:
    print("[⚠] No data collected.")
