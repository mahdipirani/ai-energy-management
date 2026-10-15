from dataclasses import dataclass
from io import StringIO
 
import numpy as np
import pandas as pd
 
# --- pars_energy/config.py : qarar-e dade, hame-chiz-e site --------
@dataclass
class SiteConfig:
    name: str = 'Pars-Ayandeh'
    tz: str = 'Asia/Tehran'
    freq: str = '15min'              # rizdangi-ye kontrar
    unit: str = 'kWh'                # vahed-e zakhire-sazi
    peak_window: tuple = (17, 23)    # panjere-ye geran
    low_window: tuple = (0, 8)       # panjere-ye arzan
 
SITE = SiteConfig()
 
# --- pars_energy/io.py : khandan + e'tebar-sanzi-ye damene ----------
def read_meter(csv_text, cfg):
    df = (pd.read_csv(StringIO(csv_text), parse_dates=['ts'],
                      index_col='ts').sort_index())
    dup = int(df.index.duplicated().sum())
    neg = int((df['kwh'] < 0).sum())
    if dup or neg:
        raise ValueError(f'duplicates={dup} negative={neg}')
    return df
 
# --- pars_energy/clean.py : gozarsh-e keyfiyat (fasl 5) -------------
def quality_report(df, cfg):
    grid = pd.date_range(df.index[0], df.index[-1], freq=cfg.freq)
    z = (df['kwh'] - df['kwh'].mean()) / df['kwh'].std()
    return pd.Series({
        'coverage': round(len(df) / len(grid), 3),
        'stuck_rows': int((df['kwh'].diff() == 0).sum()),
        'outliers_4s': int((z.abs() > 4).sum()),
    }, name=cfg.name)
 
# --- do ruz-e nemune (192 nemune-ye 15-daqiqei) ---------------------
rng = np.random.default_rng(1901)
ts = pd.date_range('2027-03-25', periods=192, freq='15min')
shape = 2.4 + 1.15 * np.clip(np.sin(np.pi * (ts.hour - 5) / 13), 0, None)
kwh = np.asarray(shape + rng.normal(0, 0.06, 192)).round(3)
kwh[:28] = 0.9                       # shab-e kam-bar-e akhar-e hafte
demo = pd.DataFrame({'ts': ts, 'kwh': kwh})
 
df = read_meter(demo.to_csv(index=False), SITE)
print('site :', SITE.name, '|', SITE.freq, '|', SITE.unit)
print(quality_report(df, SITE).to_string())
print('hourly mean (kWh):', df['kwh'].resample('h')
      .mean().round(2).head(4).tolist())
 
