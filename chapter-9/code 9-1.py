import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(901)
 
# --- daily site energy (MWh), Farvardin..Aban 1405 -------------------------
# Monthly totals are calibrated to the metering report of chapter 3.
bounds = ['2026-03-21', '2026-04-21', '2026-05-22', '2026-06-22',
          '2026-07-23', '2026-08-23', '2026-09-23', '2026-10-23',
          '2026-11-22']
targets = [528, 531, 536, 538, 543, 552, 544, 539]   # MWh per month
idx = pd.date_range('2026-03-21', '2026-11-21', freq='D')
 
month_temp = {'03': 14.0, '04': 19.5, '05': 24.5, '06': 30.0,
              '07': 32.0, '08': 31.0, '09': 27.5, '10': 20.5,
              '11': 14.5}
temp = pd.Series([month_temp[ts.strftime('%m')] + rng.normal(0, 2.5)
                  for ts in idx], index=idx)
hdd = (18 - temp).clip(lower=0).round(1)
cdd = (temp - 24).clip(lower=0).round(1)
 
def day_factor(ts):
    wd = ts.weekday()                    # Mon=0 .. Sun=6
    if wd == 4:  return 0.52             # Friday, plant idle
    if wd == 3:  return 0.88             # Thursday, half shift
    return 1.00                          # Sat..Wed, full production
 
growth = np.linspace(0, 1, len(idx))
raw = pd.Series(0.0, index=idx)
for ts in idx:
    f = day_factor(ts) * (1 + 0.018 * growth[(ts - idx[0]).days])
    raw[ts] = 19.6 * f + 0.16 * hdd[ts] + 0.07 * cdd[ts]
raw *= (1 + rng.normal(0, 0.028, len(idx)))
 
# the chapter-8 heater: +0.385 MWh/day, switched on for all of Aban
raw[raw.index >= '2026-10-23'] += 0.385
 
mwh = raw.copy()                         # exact monthly calibration
for m in range(8):
    mask = (mwh.index >= bounds[m]) & (mwh.index < bounds[m + 1])
    mwh[mask] *= targets[m] / mwh[mask].sum()
mwh = mwh.round(2)
 
print('--- series head (MWh/day) ---')
print(mwh.head(5).to_string())
print(f"days : {len(mwh)}   mean : {mwh.mean():.1f}   "
      f"std : {mwh.std():.1f}   CV : {mwh.std() / mwh.mean():.2f}")
 
names = ['Far', 'Ord', 'Kho', 'Tir', 'Mor', 'Sha', 'Mehr', 'Aban']
sums = []
for m in range(8):
    mask = (mwh.index >= bounds[m]) & (mwh.index < bounds[m + 1])
    sums.append(round(mwh[mask].sum()))
print('monthly totals (MWh) :', dict(zip(names, sums)))
 
wd_map = {5: 'Sat', 6: 'Sun', 0: 'Mon', 1: 'Tue', 2: 'Wed',
          3: 'Thu', 4: 'Fri'}
by_wd = mwh.groupby(mwh.index.weekday).mean()
print('--- weekday means (MWh/day) ---')
print({wd_map[k]: round(v, 1) for k, v in by_wd.items()})
 
fig, ax = plt.subplots(figsize=(10, 4.4))
ax.plot(mwh.index, mwh, color='#0B4E6B', lw=0.9, marker='.',
        ms=2.5, label='daily energy')
roll = mwh.rolling(7, center=True).mean()
ax.plot(roll.index, roll, color='#0E8A6A', lw=2.4,
        label='7-day rolling mean')
fri = mwh[mwh.index.weekday == 4]
ax.scatter(fri.index, fri, color='#B46C16', s=13, zorder=5,
           label='Fridays')
ax.set_xlabel('date')
ax.set_ylabel('MWh/day')
ax.set_title('Pars-Ayandeh daily electricity: the weekly heartbeat')
ax.legend(fontsize=9, loc='lower left')
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
