import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(601)          # same seed -> same daily energy
 
# --- rebuild daily furnace energy (identical to code 6-1) -----------------
bounds = ['2026-03-21', '2026-04-21', '2026-05-22', '2026-06-22',
          '2026-07-23', '2026-08-23', '2026-09-23', '2026-10-23',
          '2026-11-22']
targets = [208, 210, 214, 215, 220, 224, 221, 217]
idx = pd.date_range('2026-03-21', '2026-11-21', freq='D')
growth = np.linspace(0, 1, len(idx))
shape = pd.Series(0.0, index=idx)
for ts in idx:
    wd = ts.weekday()
    if wd == 4:    f = 0.40
    elif wd == 3:  f = 0.72
    else:          f = 1.00
    shape[ts] = f * (1 + 0.09 * growth[(ts - idx[0]).days])
daily = shape * (1 + rng.normal(0, 0.035, len(idx)))
for m in range(8):
    mask = (daily.index >= bounds[m]) & (daily.index < bounds[m + 1])
    daily[mask] *= targets[m] / daily[mask].sum()
kwh = (daily * 1000).round(0)             # kWh per day
 
# --- daily production (tons) from the ERP log ------------------------------
tons = pd.Series(0.0, index=idx)
for ts in idx:
    wd = ts.weekday()
    if wd == 4:    tons[ts] = 4
    elif wd == 3:  tons[ts] = 23
    else:          tons[ts] = 34
tons = tons * (1 + 0.05 * growth) * (1 + rng.normal(0, 0.04, len(idx)))
 
# --- daily mean temperature (deg C) -> heating degree days -----------------
month_temp = {'03': 14.0, '04': 19.5, '05': 24.5, '06': 30.0,
              '07': 32.0, '08': 31.0, '09': 27.5, '10': 20.5, '11': 14.5}
temp = pd.Series([month_temp[ts.strftime('%m')] +
                  rng.normal(0, 2.5) for ts in idx], index=idx)
hdd = (18 - temp).clip(lower=0).round(1)
 
# --- multi-driver baseline regression ---------------------------------------
X = pd.DataFrame({'tons': tons, 'hdd': hdd})
model = LinearRegression().fit(X, kwh)
kwh_hat = pd.Series(model.predict(X), index=idx)
resid = kwh - kwh_hat
print(f"baseline : kWh = {model.coef_[0]:.1f} * tons "
      f"+ {model.coef_[1]:.1f} * hdd + {model.intercept_:.0f}")
print(f"R2 = {model.score(X, kwh):.3f}   "
      f"RMSE = {np.sqrt((resid ** 2).mean()):.0f} kWh/day")
 
names = ['Far', 'Ord', 'Kho', 'Tir', 'Mor', 'Sha', 'Mehr', 'Aban']
mres = []
for m in range(8):
    mask = (resid.index >= bounds[m]) & (resid.index < bounds[m + 1])
    mres.append(round(resid[mask].mean()))
print('monthly mean residual :', dict(zip(names, mres)))
fri = resid[resid.index.weekday == 4]
work = resid[resid.index.weekday.isin([5, 6, 0, 1, 2])]
print(f"residual Friday {fri.mean():.0f} | workday {work.mean():.0f} "
      f"kWh/day")
 
# --- residual control chart ---------------------------------------------------
mu, sig = resid.mean(), resid.std()
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(resid.index, resid, color='#0B4E6B', lw=0.9, marker='.',
        ms=3, label='residual')
ax.axhline(mu, color='#0E8A6A', lw=1.6, label=f'mean {mu:.0f}')
ax.axhline(3 * sig, color='#B46C16', ls='--', lw=1.4,
           label=f'+3 sigma = {3 * sig:.0f}')
ax.axhline(-3 * sig, color='#B46C16', ls='--', lw=1.4)
out = resid.abs() > 3 * sig
ax.scatter(resid.index[out], resid[out], color='#8B2E2E', zorder=5, s=30)
ax.set_xlabel('date')
ax.set_ylabel('residual (kWh/day)')
ax.set_title('Baseline residuals: actual minus predicted furnace energy')
ax.legend(fontsize=8)
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
