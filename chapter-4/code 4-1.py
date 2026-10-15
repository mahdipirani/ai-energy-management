import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- work in the script's own folder (files saved next to the code) ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

rng = np.random.default_rng(11)

# --- the vendor portal exports one CSV per week ------------------------
# (here we create the 4 files to keep the example self-contained)
idx = pd.date_range('2026-09-07', periods=28, freq='D')
load = []
for d in idx:
    if d.weekday() == 4:                    # Friday -> plant idle
        load.append(2100 + rng.normal(0, 90))
    elif d.weekday() == 3:                  # Thursday -> half shift
        load.append(5200 + rng.normal(0, 150))
    else:
        load.append(7300 + rng.normal(0, 220))
daily = pd.DataFrame({'date': idx, 'kwh': np.round(load, 0)})

# week files: note the overlap day between weeks 1 and 2
for w in range(4):
    part = daily.iloc[w*7:(w+1)*7 + (1 if w == 0 else 0)]
    part.to_csv(f'meter_furnace_w{w+1}.csv', index=False)
print('files:', sorted(glob.glob('meter_furnace_w*.csv')))

# --- 1) read all weekly files into one DataFrame -----------------------
frames = []
for path in sorted(glob.glob('meter_furnace_w*.csv')):
    part = pd.read_csv(path, parse_dates=['date'])
    part['source'] = path                       # keep lineage
    frames.append(part)
merged = pd.concat(frames, ignore_index=True)
print(f"rows after concat     : {len(merged)}")

# --- 2) deduplicate, sort, validate ------------------------------------
merged = (merged.drop_duplicates(subset='date')
                 .sort_values('date')
                 .reset_index(drop=True))
merged['kw_avg'] = (merged['kwh'] / 24).round(1)
print(f"rows after dedup      : {len(merged)}")
print(f"duplicate rows removed: 1 (overlap 2026-09-14)")
print(merged.head(6).to_string(index=False))

# --- 3) store once, in an efficient format ------------------------------
merged.to_parquet('furnace_daily.parquet', index=False)
back = pd.read_parquet('furnace_daily.parquet')
print('parquet round-trip OK :', len(back) == len(merged))

plt.figure(figsize=(10, 4.0))
plt.bar(merged['date'], merged['kwh'] / 1000, color='#0B4E6B',
        width=0.8)
plt.axhline(merged['kwh'].mean() / 1000, color='#B46C16', ls='--',
            lw=1.6, label='mean')
plt.ylabel('energy (MWh/day)')
plt.title('Furnace sub-meter: 28 days after pipeline merge')
plt.legend()
plt.grid(alpha=0.25, ls=':', axis='y')
ax = plt.gca()    
ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))   
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d')) 
plt.xticks(rotation=45, ha='right') 

plt.tight_layout()
plt.show()