import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# --- pars_energy/kpi.py : shakhes-ha-ye ruzane ----------------------
def daily_kpi(df):
    kwh = df['kwh'].resample('D').sum()
    peak_kw = df['kwh'].resample('D').max() * 4   # kWh/15min -> kW
    out = pd.DataFrame({'kwh': kwh, 'peak_kw': peak_kw})
    out['load_factor'] = out['kwh'] / 24 / out['peak_kw']
    return out
 
# --- pars_energy/report.py : gozarsh-e yek-safhei-ye mahane ---------
def monthly_report(df, tons_month, low_window=(0, 8)):
    k = daily_kpi(df)
    hourly = df['kwh'].resample('h').sum()
    low = hourly[hourly.index.hour < low_window[1]].sum()
    return {'total_kwh': k['kwh'].sum(),
            'enpi': k['kwh'].sum() / tons_month,
            'load_factor': k['load_factor'].mean(),
            'peak_kw': k['peak_kw'].max(),
            'low_share': low / hourly.sum()}
 
# --- dade-ye mah-e nemune: 30 ruz, rizdangi 15min -------------------
rng = np.random.default_rng(1902)
ts = pd.date_range('2027-03-21', periods=30 * 96, freq='15min')
shape = 70 + 330 * np.clip(np.sin(np.pi * (ts.hour - 5) / 13), 0, None)
shape = np.where(ts.dayofweek >= 6, shape * 0.55, shape)
kwh = (shape + rng.normal(0, 9, len(ts))).clip(1).round(2)
df = pd.DataFrame({'kwh': kwh}, index=ts)
 
r = monthly_report(df, tons_month=879)   # tolid-e mahane (ton)
print('=== Pars-Ayandeh — monthly one-page report (Far 1407) ===')
print(f"energy      : {r['total_kwh']/1000:.1f} MWh"
      f"    EnPI : {r['enpi']/1000:.3f} MWh/ton")
print(f"load factor : {r['load_factor']:.2f}"
      f"       peak  : {r['peak_kw']:.0f} kW")
print(f"low-window share (0-8): {100*r['low_share']:.1f} %"
      f"   target: >= 18 %")
 
# --- gha'ede-ye tasmim (haman-e jalse) ------------------------------
if r['low_share'] < 0.18:
    print('decision    : move flexible load (air charge) to 0-8')
elif r['load_factor'] < 0.55:
    print('decision    : investigate peak shaving')
else:
    print('decision    : no action — thresholds not triggered')
 
# --- nemudar-e do-paneli -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
k = daily_kpi(df)
axes[0].bar(k.index.day, k['kwh'] / 1000, color='#0B4E6B', alpha=0.8)
axes[0].plot(k.index.day, (k['kwh'].rolling(7).mean() / 1000),
             color='#0E8A6A', lw=2.2, label='7-day rolling')
axes[0].set_xlabel('day of month')
axes[0].set_ylabel('MWh / day')
axes[0].set_title('daily energy')
axes[0].legend(fontsize=8.5)
axes[0].grid(alpha=0.25, ls=':')
prof = df['kwh'].groupby(df.index.hour).mean()
axes[1].plot(prof.index, prof, color='#0B4E6B', lw=2.2, marker='o', ms=3)
axes[1].axvspan(0, 8, color='#0E8A6A', alpha=0.12, label='low window')
axes[1].axvspan(17, 23, color='#B46C16', alpha=0.12, label='peak window')
axes[1].set_xlabel('hour of day')
axes[1].set_ylabel('mean kWh / 15min')
axes[1].set_title('load profile')
axes[1].legend(fontsize=8.5)
axes[1].grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
