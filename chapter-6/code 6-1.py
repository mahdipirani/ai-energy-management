import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(601)
 
# --- 8 months of daily furnace energy (Farvardin..Aban 1405) -------------
# Persian month boundaries (Gregorian); targets from the sub-meter report
bounds = ['2026-03-21', '2026-04-21', '2026-05-22', '2026-06-22',
          '2026-07-23', '2026-08-23', '2026-09-23', '2026-10-23',
          '2026-11-22']
targets = [208, 210, 214, 215, 220, 224, 221, 217]        # MWh / month
idx = pd.date_range('2026-03-21', '2026-11-21', freq='D')  # 246 days
 
# weekday shape (Sat..Wed full, Thu half, Fri idle) + slow growth
growth = np.linspace(0, 1, len(idx))
shape = pd.Series(0.0, index=idx)
for ts in idx:
    wd = ts.weekday()                     # Mon=0 .. Sun=6
    if wd == 4:    f = 0.40               # Friday, plant idle
    elif wd == 3:  f = 0.72               # Thursday, half shift
    else:          f = 1.00               # Saturday..Wednesday
    shape[ts] = f * (1 + 0.09 * growth[(ts - idx[0]).days])
noise = rng.normal(0, 0.035, len(idx))
daily = shape * (1 + noise)
 
# scale each Persian month to the sub-meter totals exactly
for m in range(8):
    mask = (daily.index >= bounds[m]) & (daily.index < bounds[m + 1])
    daily[mask] *= targets[m] / daily[mask].sum()
daily = daily.round(1)                    # MWh per day
 
# --- 1) descriptive statistics -------------------------------------------
print('--- describe (MWh/day) ---')
print(daily.describe().round(2).to_string())
print('CV = std/mean :', round(daily.std() / daily.mean(), 3))
 
# --- 2) monthly picture ---------------------------------------------------
names = ['Far', 'Ord', 'Kho', 'Tir', 'Mor', 'Sha', 'Mehr', 'Aban']
rows = []
for m in range(8):
    seg = daily[(daily.index >= bounds[m]) & (daily.index < bounds[m + 1])]
    rows.append(dict(month=names[m], mwh=round(seg.sum(), 1),
                     day_avg=round(seg.mean(), 2),
                     cv=round(seg.std() / seg.mean(), 3)))
monthly = pd.DataFrame(rows)
print('--- monthly ---')
print(monthly.to_string(index=False))
 
# --- 3) calendar effect: weekday ------------------------------------------
wd_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri',
            5: 'Sat', 6: 'Sun'}
g = daily.groupby(daily.index.weekday).mean()
g.index = [wd_names[i] for i in g.index]
week = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']
print('--- weekday (MWh/day) ---')
print(g.reindex(week).round(2).to_string())
 
# --- 4) key findings ------------------------------------------------------
work = daily[daily.index.weekday.isin([5, 6, 0, 1, 2])].mean()
fri = daily[daily.index.weekday == 4].mean()
thu = daily[daily.index.weekday == 3].mean()
print(f"peak day   : {daily.max():.1f} MWh on {daily.idxmax().date()}")
print(f"lowest day : {daily.min():.1f} MWh on {daily.idxmin().date()}")
print(f"Fri mean {fri:.2f} | Thu {thu:.2f} | workday {work:.2f} MWh "
      f"-> Friday = {100*fri/work:.0f} % of workday")
 
# --- 5) figures ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.bar(daily.index, daily, color='#9CC8D8', width=0.85)
ax.plot(daily.index, daily.rolling(7, center=True).mean(),
        color='#0B4E6B', lw=2, label='7-day rolling mean')
mm = [daily[(daily.index >= bounds[m]) &
            (daily.index < bounds[m + 1])].mean() for m in range(8)]
mid = [pd.Timestamp(bounds[m]) +
       (pd.Timestamp(bounds[m + 1]) - pd.Timestamp(bounds[m])) / 2
       for m in range(8)]
ax.plot(mid, mm, color='#B46C16', lw=1.6, ls='--', marker='s', ms=4,
        label='monthly mean')
ax.set_xlabel('date (Farvardin..Aban 1405)')
ax.set_ylabel('furnace energy (MWh/day)')
ax.set_title('Daily furnace energy: raw, rolling mean, monthly mean')
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.25, ls=':', axis='y')
plt.tight_layout()
plt.show()
 
fig, ax = plt.subplots(figsize=(10, 4.2))
by_month = [daily[(daily.index >= bounds[m]) &
                  (daily.index < bounds[m + 1])].values for m in range(8)]
ax.boxplot(by_month)
ax.set_xticklabels(names)
ax.set_xlabel('Persian month of 1405')
ax.set_ylabel('MWh/day')
ax.set_title('Distribution of daily furnace energy by month')
ax.grid(alpha=0.25, ls=':', axis='y')
plt.tight_layout()
plt.show()
 

