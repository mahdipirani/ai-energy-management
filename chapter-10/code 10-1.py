import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(901)   # same series as code 9-1
 
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
 
test = mwh[mwh.index >= '2026-10-23']       # 30 Aban days
 
def metrics(y, yhat):
    err = (y - yhat).abs()
    mae = err.mean()
    rmse = np.sqrt(((y - yhat) ** 2).mean())
    mape = 100 * (err / y).mean()
    return round(mae, 2), round(rmse, 2), round(mape, 1)
 
models = {
    'yesterday (lag 1)': mwh.shift(1),
    'same weekday last week (lag 7)': mwh.shift(7),
    'mean of last 7 days': mwh.shift(1).rolling(7).mean(),
}
print('--- naive baselines, 30 Aban days ---')
for name, series in models.items():
    mae, rmse, mape = metrics(test, series[test.index])
    print(f"{name:32s} MAE {mae:5.2f}  RMSE {rmse:5.2f}  "
          f"MAPE {mape:4.1f} %")
 
wd = test.index.weekday
best = models['same weekday last week (lag 7)'][test.index]
print('--- MAPE of the lag-7 guess, by day type ---')
for label, mask in [('workday (Sat-Wed)', wd.isin([5, 6, 0, 1, 2])),
                    ('Thursday', wd == 3),
                    ('Friday', wd == 4)]:
    mape = 100 * ((test[mask] - best[mask]).abs()
                  / test[mask]).mean()
    print(f"{label:20s} : {mape:.1f} %")
 
fig, ax = plt.subplots(figsize=(10, 4.2))
w = test.iloc[-10:]
ax.plot(w.index, w, color='#0B4E6B', lw=2, marker='o', ms=4.5,
        label='actual')
ax.plot(w.index, best[w.index], color='#B46C16', lw=1.4, ls='--',
        marker='s', ms=3.5, label='same weekday, last week')
ax.set_xlabel('date')
ax.set_ylabel('MWh/day')
ax.set_title('Aban: actual versus the best naive guess')
ax.legend(fontsize=9)
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
