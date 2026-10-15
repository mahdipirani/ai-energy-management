import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(901)   # same seed -> same series as 9-1
 
# --- rebuild the daily series (identical to code 9-1) ----------------------
bounds = ['2026-03-21', '2026-04-21', '2026-05-22', '2026-06-22',
          '2026-07-23', '2026-08-23', '2026-09-23', '2026-10-23',
          '2026-11-22']
targets = [528, 531, 536, 538, 543, 552, 544, 539]
idx = pd.date_range('2026-03-21', '2026-11-21', freq='D')
month_temp = {'03': 14.0, '04': 19.5, '05': 24.5, '06': 30.0,
              '07': 32.0, '08': 31.0, '09': 27.5, '10': 20.5,
              '11': 14.5}
temp = pd.Series([month_temp[ts.strftime('%m')] + rng.normal(0, 2.5)
                  for ts in idx], index=idx)
hdd = (18 - temp).clip(lower=0).round(1)
cdd = (temp - 24).clip(lower=0).round(1)
 
def day_factor(ts):
    wd = ts.weekday()
    if wd == 4:  return 0.52
    if wd == 3:  return 0.88
    return 1.00
 
growth = np.linspace(0, 1, len(idx))
raw = pd.Series(0.0, index=idx)
for ts in idx:
    f = day_factor(ts) * (1 + 0.018 * growth[(ts - idx[0]).days])
    raw[ts] = 19.6 * f + 0.16 * hdd[ts] + 0.07 * cdd[ts]
raw *= (1 + rng.normal(0, 0.028, len(idx)))
raw[raw.index >= '2026-10-23'] += 0.385
mwh = raw.copy()
for m in range(8):
    mask = (mwh.index >= bounds[m]) & (mwh.index < bounds[m + 1])
    mwh[mask] *= targets[m] / mwh[mask].sum()
mwh = mwh.round(2)
 
# --- 1) feature engineering --------------------------------------------------
df = pd.DataFrame({'mwh': mwh})
df['lag_1'] = df['mwh'].shift(1)
df['lag_7'] = df['mwh'].shift(7)
df['roll_7'] = df['mwh'].shift(1).rolling(7).mean()
df['hdd'] = hdd
df['cdd'] = cdd
df['is_friday'] = (df.index.weekday == 4).astype(int)
df['is_thursday'] = (df.index.weekday == 3).astype(int)
df['t'] = np.arange(len(df)) / len(df)
df = df.dropna()
print('--- feature table (a Friday and its neighbours) ---')
print(df.loc['2026-10-15':'2026-10-18',
             ['mwh', 'lag_1', 'lag_7', 'roll_7', 'hdd',
              'is_friday']].round(2).to_string())
 
# --- 2) autocorrelation: the memory of the series -----------------------------
acf = pd.Series({k: df['mwh'].autocorr(k) for k in range(1, 15)})
print('--- autocorrelation by lag (days) ---')
print({k: round(v, 2) for k, v in acf.items()})
 
fig, ax = plt.subplots(figsize=(10, 3.8))
colors = ['#B46C16' if k % 7 == 0 else '#0B4E6B' for k in acf.index]
ax.bar(acf.index, acf.values, color=colors, width=0.62)
band = 1.96 / np.sqrt(len(df))
ax.axhspan(-band, band, color='#0E8A6A', alpha=0.15)
ax.axhline(0, color='#475569', lw=0.8)
ax.set_xlabel('lag (days)')
ax.set_ylabel('autocorrelation')
ax.set_title('Memory of the series: weekly echoes at lags 7 and 14')
ax.set_xticks(range(1, 15))
ax.grid(alpha=0.25, ls=':', axis='y')
plt.tight_layout()
plt.show()
 
# --- 3) the only legal split: cut in time -------------------------------------
train = df[df.index < '2026-10-23']       # Farvardin .. end of Mehr
test = df[df.index >= '2026-10-23']        # Aban, the exam month
print(f"train : {len(train)} days "
      f"({train.index[0].date()} .. {train.index[-1].date()})")
print(f"test  : {len(test)} days (all of Aban, kept untouched)")
print(f"target mean  train {train['mwh'].mean():.1f} | "
      f"test {test['mwh'].mean():.1f} MWh/day")
 
