import json
import numpy as np
import pandas as pd
 
# --- a function that mimics a REST API call ----------------------------
def fetch_day(device_id, date_str):
    """Return JSON text of 15-minute records for one day.
 
    In production this body becomes:
        import requests
        r = requests.get(f"{BASE_URL}/devices/{device_id}/data",
                         params={"date": date_str,
                                 "resolution": "15min"},
                         headers={"Authorization": f"Bearer {TOKEN}"},
                         timeout=10)
        r.raise_for_status()
        return r.text
    """
    day = int(date_str[-2:])
    rng = np.random.default_rng(1000 + day)     # deterministic
    records = []
    for slot in range(96):
        h = slot / 4
        if 6 <= h < 14:      kw = 620 + rng.normal(0, 30)
        elif 14 <= h < 22:   kw = 480 + rng.normal(0, 20)
        else:                kw = 175 + rng.normal(0, 10)
        records.append(dict(
            device=str(device_id), date=date_str, slot=slot,
            ts=f"{date_str}T{int(h):02d}:{15*(slot%4):02d}:00",
            kw=round(float(kw), 1)))
    return json.dumps(records)
 
# --- call the "API" for 7 days and normalize ---------------------------
frames = []
for day in range(14, 21):                       # 2026-09-14 .. 20
    date_str = f"2026-09-{day:02d}"
    raw = fetch_day('FUR-01', date_str)         # furnace meter
    frames.append(pd.json_normalize(json.loads(raw)))
api = pd.concat(frames, ignore_index=True)
api['ts'] = pd.to_datetime(api['ts'])
 
print(f"records fetched : {len(api)}")
print(api[['ts', 'kw']].head(5).to_string(index=False))
print('---')
print(api['kw'].describe().round(1).to_string())
 
# --- 15-minute time series, ready for analysis -------------------------
ts = (api.set_index('ts')['kw']
         .sort_index()
         .asfreq('15min'))
print(f"15-min slots    : {len(ts)}  "
      f"(expected 672 -> gaps: {672 - ts.notna().sum()})")
 
