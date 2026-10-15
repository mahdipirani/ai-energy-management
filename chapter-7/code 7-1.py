import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(701)
 
# --- main-meter load, 15-min, one month (Mehr 1405) ----------------------
idx = pd.date_range('2026-09-23', periods=30 * 96, freq='15min')
 
def site_kw(ts):
    h = ts.hour + ts.minute / 60
    wd = ts.weekday()                     # Mon=0 .. Sun=6
    if wd == 4:                                       # Friday, plant idle
        return 400 + 25 * np.sin((h - 6) * np.pi / 12) ** 2
    if wd == 3:                                       # Thursday, half shift
        if h < 7:   return 520
        if h < 13:  return 1000
        if h < 22:  return 610
        return 540
    if h < 6:    return 520                           # Sat..Wed night
    if h < 7:    return 520 + 600 * (h - 6)           # morning ramp
    if h < 16:   return 1120                          # full production
    if h < 18:   return 900
    if h < 22:   return 700
    return 560
 
kw = pd.Series([site_kw(ts) for ts in idx], index=idx)
kw = (kw * (1 + rng.normal(0, 0.03, len(kw)))).round(0)
print(f"month energy   : {kw.sum() / 4 / 1000:.0f} MWh")
 
# --- 1) average day profile by day type ----------------------------------
def day_type(ts):
    wd = ts.weekday()
    if wd == 4:  return 'Friday'
    if wd == 3:  return 'Thursday'
    return 'Workday (Sat-Wed)'
 
prof = (kw.groupby([kw.index.hour, kw.index.map(day_type)])
          .mean().unstack())
print('--- average profile (kW) ---')
print(prof.loc[[3, 7, 12, 17, 21]].round(0).to_string())
 
fig, ax = plt.subplots(figsize=(10, 4.2))
styles = {'Workday (Sat-Wed)': ('#0B4E6B', '-'),
          'Thursday': ('#B46C16', '--'),
          'Friday': ('#0E8A6A', ':')}
for col in prof.columns:
    c, ls = styles[col]
    ax.plot(prof.index, prof[col], color=c, ls=ls, lw=2.2, label=col)
ax.set_xlabel('hour of day')
ax.set_ylabel('average load (kW)')
ax.set_title('Average day profiles by day type (Mehr 1405)')
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=9)
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
# --- 2) key indicators -----------------------------------------------------
base = kw.quantile(0.05)
peak = kw.max()
print(f"base load (P5) : {base:.0f} kW")
print(f"peak load      : {peak:.0f} kW")
print(f"load factor    : {kw.mean() / peak:.2f}")
 
# --- 3) load duration curve -------------------------------------------------
s = kw.sort_values(ascending=False).values
hours = np.arange(len(s)) / 4
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(hours, s, color='#0B4E6B', lw=2)
ax.axhline(base, color='#0E8A6A', ls='--', lw=1.5,
           label=f'base load ~{base:.0f} kW')
ax.axvspan(0, 0.05 * len(s) / 4, color='#B46C16', alpha=0.18)
ax.text(0.09 * len(s) / 4 + 10, kw.quantile(0.93), 'top 5% hours',
        color='#B46C16', fontsize=9)
ax.set_xlabel('hours of month, sorted from high to low')
ax.set_ylabel('load (kW)')
ax.set_title('Load duration curve (Mehr 1405)')
ax.legend(fontsize=9)
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
# --- 4) time-of-use shares (tariff of chapter 3) ----------------------------
def period(ts):
    if ts.hour < 7:        return 'off-peak'
    if 17 <= ts.hour < 23: return 'peak'
    return 'mid'
 
tariff = {'off-peak': 18.0, 'mid': 32.0, 'peak': 65.0}  # M-IRR / MWh
tou = kw.groupby(kw.index.map(period)).agg(kwh=lambda v: v.sum() / 4)
tou['energy_pct'] = 100 * tou['kwh'] / tou['kwh'].sum()
tou['tariff'] = pd.Series(tariff)
tou['cost'] = tou['kwh'] / 1000 * tou['tariff']
tou['cost_pct'] = 100 * tou['cost'] / tou['cost'].sum()
print('--- time-of-use ---')
print(tou.round(1).to_string())
 
