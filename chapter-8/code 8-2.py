import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(802)
 
# --- daily unaccounted energy = main meter - all sub-meters ----------------
# Monthly totals follow the sub-meter report of chapter 3:
#   months 1..5 : commissioning of the new sub-meters (coverage 92->99 %)
#   months 6..7 : stable, ~4 MWh/month of normal metering error
#   month 8     : a hidden load appears from Aban 1st (2026-10-23)
bounds = ['2026-03-21', '2026-04-21', '2026-05-22', '2026-06-22',
          '2026-07-23', '2026-08-23', '2026-09-23', '2026-10-23',
          '2026-11-22']
idx = pd.date_range('2026-03-21', '2026-11-21', freq='D')
 
base_level = [1.30, 1.16, 0.84, 0.58, 0.19, 0.10, 0.135, 0.135]
unacc = pd.Series(0.0, index=idx)
for ts in idx:
    m = sum(ts >= pd.Timestamp(b) for b in bounds[1:-1])
    unacc[ts] = base_level[m] + rng.normal(0, 0.05)
 
# the hidden load: +15 kW around the clock from the first cold night
heater_start = pd.Timestamp('2026-10-23')
unacc[unacc.index >= heater_start] += 15 * 24 / 1000      # 0.36 MWh/day
 
# calibrate each Persian month to the report totals exactly
targets = [40, 36, 26, 18, 6, 3, 4, 15]
for m in range(8):
    mask = (unacc.index >= bounds[m]) & (unacc.index < bounds[m + 1])
    unacc[mask] *= targets[m] / unacc[mask].sum()
unacc = unacc.round(3)
 
# --- 1) baseline period: months 6-7, after commissioning -------------------
base_mask = (unacc.index >= '2026-08-23') & (unacc.index < '2026-10-23')
mu, sig = unacc[base_mask].mean(), unacc[base_mask].std()
ucl = mu + 3 * sig
print(f"baseline unaccounted : {mu:.3f} +- {sig:.3f} MWh/day "
      f"(UCL = {ucl:.3f})")
 
aban = unacc[unacc.index >= '2026-10-23']
flagged = aban[aban > ucl]
print(f"Aban days over UCL   : {len(flagged)} of {len(aban)}")
print(f"first flagged day    : {flagged.index.min().date()}")
 
# --- 2) size the hidden load -------------------------------------------------
excess_day = aban.mean() - mu
excess_mwh = excess_day * len(aban)
hidden_kw = excess_day * 1000 / 24
print(f"excess               : {excess_mwh:.1f} MWh in {len(aban)} days")
print(f"implied hidden load  : {hidden_kw:.1f} kW continuous")
 
# --- 3) control chart ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(unacc.index, unacc, color='#0B4E6B', lw=1, marker='.', ms=3)
ax.axhline(mu, color='#0E8A6A', lw=1.6,
           label=f'baseline {mu:.2f} MWh/day')
ax.axhline(ucl, color='#B46C16', ls='--', lw=1.5,
           label=f'UCL {ucl:.2f}')
ax.scatter(flagged.index, flagged.values, color='#8B2E2E', s=26,
           zorder=5)
ax.axvline(heater_start, color='#8B2E2E', lw=1.2, ls=':')
ax.text(pd.Timestamp('2026-10-25'), ucl + 0.06,
        'Aban 1: first cold night', color='#8B2E2E', fontsize=8)
ax.set_xlabel('date')
ax.set_ylabel('MWh/day')
ax.set_title('Daily unaccounted energy: main meter minus sub-meters')
ax.legend(fontsize=8)
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
# --- 4) CUSUM of deviations -----------------------------------------------------
cusum = (unacc - mu).cumsum()
aban_gain = cusum.iloc[-1] - cusum.asof(heater_start -
                                         pd.Timedelta(days=1))
fig, ax = plt.subplots(figsize=(10, 4.0))
ax.plot(cusum.index, cusum, color='#B46C16', lw=1.8)
ax.axvline(heater_start, color='#8B2E2E', lw=1.2, ls=':')
ax.text(pd.Timestamp('2026-04-20'), cusum.max() * 0.80,
        'commissioning: meters incomplete', color='#475569',
        fontsize=8.5)
ax.text(pd.Timestamp('2026-08-10'), cusum.max() * 0.55,
        'flat: metering under control', color='#475569', fontsize=8.5)
ax.text(heater_start - pd.Timedelta(days=5),
        cusum.max() * 0.10,
        f'new climb: +{aban_gain:.1f} MWh',
        color='#8B2E2E', fontsize=8.5,
        ha='right', va='center')
ax.set_xlabel('date')
ax.set_ylabel('CUSUM (MWh)')
ax.set_title('CUSUM of deviations from the baseline')
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
# --- 5) monthly table (matches chapter 3) ----------------------------------------
names = ['Far', 'Ord', 'Kho', 'Tir', 'Mor', 'Sha', 'Mehr', 'Aban']
rows = []
for m in range(8):
    seg = unacc[(unacc.index >= bounds[m]) &
                (unacc.index < bounds[m + 1])]
    rows.append(dict(month=names[m], unaccounted=round(seg.sum(), 1)))
print('--- monthly unaccounted (MWh) ---')
print(pd.DataFrame(rows).to_string(index=False))
print(f"CUSUM gain in Aban  : {aban_gain:.1f} MWh "
      f"(total incl. commissioning: {cusum.iloc[-1]:.1f})")
 
