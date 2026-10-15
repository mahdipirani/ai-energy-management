import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
 
# --- features (identical to code 9-2) ----------------------------------------
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
FEATS = ['lag_1', 'lag_7', 'roll_7', 'hdd', 'cdd', 'is_friday',
         'is_thursday', 't']
 
# --- walk-forward evaluation: one honest re-fit every day ---------------------
dates = df.index[df.index >= '2026-10-23']
pred_lr, pred_rf = [], []
for d in dates:
    hist = df[df.index < d]                # only the past, every day
    x, y = hist[FEATS], hist['mwh']
    row = df.loc[[d], FEATS]
    pred_lr.append(LinearRegression().fit(x, y).predict(row)[0])
    rf = RandomForestRegressor(n_estimators=250, random_state=7,
                               n_jobs=-1).fit(x, y)
    pred_rf.append(rf.predict(row)[0])
 
pred_lr = pd.Series(pred_lr, index=dates).round(2)
pred_rf = pd.Series(pred_rf, index=dates).round(2)
y = df.loc[dates, 'mwh']
 
def metrics(y, yhat):
    err = (y - yhat).abs()
    return (round(float(err.mean()), 2),
            round(float(np.sqrt(((y - yhat) ** 2).mean())), 2),
            round(float(100 * (err / y).mean()), 1))
 
print('--- walk-forward on Aban (30 days): MAE, RMSE, MAPE ---')
print('linear regression :', metrics(y, pred_lr))
print('random forest     :', metrics(y, pred_rf))
 
train = df[df.index < '2026-10-23']
lr = LinearRegression().fit(train[FEATS], train['mwh'])
print('--- linear coefficients (MWh/day per unit) ---')
for f, c in sorted(zip(FEATS, lr.coef_), key=lambda z: -abs(z[1])):
    print(f"{f:13s} {c:+.3f}")
 
rf = RandomForestRegressor(n_estimators=250, random_state=7,
                           n_jobs=-1).fit(train[FEATS], train['mwh'])
imp = sorted([[f, round(float(v), 2)] for f, v in
              zip(FEATS, rf.feature_importances_)],
             key=lambda z: -z[1])
print('--- random forest feature importances ---')
print(imp)
 
fig, ax = plt.subplots(figsize=(10, 4.4))
w = y.iloc[-20:]
ax.plot(w.index, w, color='#0B4E6B', lw=2, marker='o', ms=4,
        label='actual')
ax.plot(w.index, pred_lr.iloc[-20:], color='#B46C16', ls='--',
        lw=1.4, label='linear regression')
ax.plot(w.index, pred_rf.iloc[-20:], color='#0E8A6A', ls=':',
        lw=1.9, label='random forest')
ax.set_xlabel('date')
ax.set_ylabel('MWh/day')
ax.set_title('Walk-forward daily forecast: last 20 days of Aban')
ax.legend(fontsize=9)
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
