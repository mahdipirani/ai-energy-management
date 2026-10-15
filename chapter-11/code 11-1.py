import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(1101)
rng_fc = np.random.default_rng(1102)      # weather-forecast errors
 
# --- hourly site load (kW), Far..Aban 1405 ----------------------------------
# Same calendar and monthly totals as chapters 3 and 9.
bounds = ['2026-03-21', '2026-04-21', '2026-05-22', '2026-06-22',
          '2026-07-23', '2026-08-23', '2026-09-23', '2026-10-23',
          '2026-11-22']
targets = [528, 531, 536, 538, 543, 552, 544, 539]   # MWh per month
days = pd.date_range('2026-03-21', '2026-11-21', freq='D')
idx = pd.date_range('2026-03-21', '2026-11-21 23:00', freq='h')
 
month_temp = {'03': 14.0, '04': 19.5, '05': 24.5, '06': 30.0,
              '07': 32.0, '08': 31.0, '09': 27.5, '10': 20.5,
              '11': 14.5}
temp = pd.Series([month_temp[d.strftime('%m')] + rng.normal(0, 2.5)
                  for d in days], index=days)
hdd_d = (18 - temp).clip(lower=0)
cdd_d = (temp - 24).clip(lower=0)
hdd_h = hdd_d.reindex(idx, method='ffill')   # daily -> hourly
cdd_h = cdd_d.reindex(idx, method='ffill')
 
def site_kw(ts):
    h, wd = ts.hour, ts.weekday()
    if wd == 4:                                  # Friday, plant idle
        return 410 + 20 * np.sin((h - 6) * np.pi / 12) ** 2
    if wd == 3:                                  # Thursday, half shift
        if h < 7:   return 520
        if h < 13:  return 1000
        if h < 22:  return 610
        return 540
    if h < 6:  return 520                        # Sat..Wed
    if h < 7:  return 520 + 600 * (h - 6)
    if h < 16: return 1120
    if h < 18: return 900
    if h < 22: return 700
    return 560
 
kw = pd.Series([site_kw(ts) + 20 * hdd_h[ts] + 2 * cdd_h[ts]
                for ts in idx], index=idx)
kw *= (1 + rng.normal(0, 0.022, len(kw)))
kw[kw.index >= '2026-10-23'] += 15       # the chapter-8 heater
for m in range(8):
    mask = (kw.index >= bounds[m]) & (kw.index < bounds[m + 1])
    kw[mask] *= targets[m] * 1000 / kw[mask].sum()
mehr = kw[(kw.index >= '2026-09-23') & (kw.index < '2026-10-23')]
print(f"Mehr energy check : {mehr.sum() / 1000:.0f} MWh (report: 544)")
 
# --- feature table ------------------------------------------------------------
df = pd.DataFrame({'kw': kw})
df['hour'] = df.index.hour
df['is_friday'] = (df.index.weekday == 4).astype(int)
df['is_thursday'] = (df.index.weekday == 3).astype(int)
df['lag_24'] = df['kw'].shift(24)
df['lag_168'] = df['kw'].shift(168)
df['roll_24'] = df['kw'].shift(24).rolling(24).mean()
df['hdd'] = [hdd_h[ts] for ts in df.index]
df['cdd'] = [cdd_h[ts] for ts in df.index]
df = df.dropna()
FEATS = ['hour', 'is_friday', 'is_thursday', 'lag_24', 'lag_168',
         'roll_24', 'hdd', 'cdd']
 
# the weather service: tomorrow's temperature with a realistic error
temp_fc = temp + rng_fc.normal(0, 1.2, len(temp))
hdd_fc = (18 - temp_fc).clip(lower=0)
cdd_fc = (temp_fc - 24).clip(lower=0)
 
def day_features(D, hdd_val, cdd_val):
    """24 feature rows for day D, built only from data before D."""
    rows = []
    for h in range(24):
        ts = pd.Timestamp(D) + pd.Timedelta(hours=h)
        rows.append(dict(
            hour=h,
            is_friday=int(D.weekday() == 4),
            is_thursday=int(D.weekday() == 3),
            lag_24=kw[ts - pd.Timedelta(hours=24)],
            lag_168=kw[ts - pd.Timedelta(hours=168)],
            roll_24=kw[ts - pd.Timedelta(hours=47):
                       ts - pd.Timedelta(hours=24)].mean(),
            hdd=hdd_val, cdd=cdd_val))
    return pd.DataFrame(rows)[FEATS]
 
def fit_model(until_D):
    hist = df[df.index < until_D]
    return RandomForestRegressor(n_estimators=200, random_state=7,
                                 n_jobs=-1).fit(hist[FEATS],
                                                hist['kw'])
 
# --- validation window: honest day-ahead residuals ----------------------------
val_days = pd.date_range('2026-10-25', '2026-11-14', freq='D')
res_val = []
for D in val_days:
    model = fit_model(D)
    yhat = model.predict(day_features(D, hdd_d[D], cdd_d[D]))
    act = kw.loc[D:D + pd.Timedelta(hours=23)]
    res_val.extend((yhat - act).values)
res_val = np.array(res_val)
q05, q95 = np.quantile(res_val, [0.05, 0.95])
print(f"validation residuals : mean {res_val.mean():.0f} kW, "
      f"90% band [{q05:.0f}, {q95:.0f}] kW")
 
# --- test week: forecast weather vs perfect weather -----------------------------
test_days = pd.date_range('2026-11-15', '2026-11-21', freq='D')
err_p, err_f, block_err = [], [], {}
blocks = {'night 00-07': range(0, 7), 'day 07-17': range(7, 17),
          'evening 17-23 (peak tariff)': range(17, 23)}
preds_fc = {}
for D in test_days:
    model = fit_model(D)
    pred_p = model.predict(day_features(D, hdd_d[D], cdd_d[D]))
    pred_f = model.predict(day_features(D, hdd_fc[D], cdd_fc[D]))
    hours = pd.date_range(D, periods=24, freq='h')
    preds_fc[D] = pd.Series(pred_f, index=hours)
    act = kw.loc[hours].values
    err_p.extend(np.abs(pred_p - act) / act)
    err_f.extend(np.abs(pred_f - act) / act)
    for label, hrs in blocks.items():
        block_err.setdefault(label, []).extend(
            np.abs(pred_f[h] - act[h]) / act[h] for h in hrs)
print(f"test MAPE, perfect weather  : {100 * np.mean(err_p):.1f} %")
print(f"test MAPE, forecast weather : {100 * np.mean(err_f):.1f} %")
print('--- MAPE by block (forecast weather) ---')
for label, vals in block_err.items():
    print(f"{label:28s} : {100 * np.mean(vals):.1f} %")
 
# --- showcase day: Aban 30 ------------------------------------------------------
D = pd.Timestamp('2026-11-21')
hours = pd.date_range(D, periods=24, freq='h')
act = kw.loc[hours]
pred = preds_fc[D]
fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(24)
ax.plot(x, act.values, color='#0B4E6B', lw=2.2, marker='o', ms=4,
        label='actual')
ax.plot(x, pred.values, color='#0E8A6A', lw=2, ls='--',
        label='day-ahead forecast')
ax.fill_between(x, pred + q05, pred + q95, color='#0E8A6A',
                alpha=0.15, label='90% prediction band')
ax.axvspan(17, 20, color='#B46C16', alpha=0.08)
ax.text(17.2, act.max() * 0.94, 'peak tariff 17-23', color='#B46C16',
        fontsize=8.5)
ax.set_xlabel('hour of day')
ax.set_ylabel('load (kW)')
ax.set_title('Day-ahead hourly forecast, Aban 30 '
             '(weather-forecast error included)')
ax.set_xticks(range(0, 24, 2))
ax.legend(fontsize=9)
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
# --- first operational forecast: Azar 1 -----------------------------------------
D1 = pd.Timestamp('2026-11-22')
az1_temp = 12.5                       # tomorrow, from the weather service
model = fit_model(D1)
yhat = model.predict(day_features(D1, 18 - az1_temp, 0))
daily = yhat.sum() / 1000
print('--- first operational forecast: Azar 1 ---')
print(f"daily energy : {daily:.1f} MWh")
print(f"peak         : {yhat.max():.0f} kW at "
      f"{int(np.argmax(yhat)):02d}:00")
print('selected hours (kW) :',
      {h: round(yhat[h]) for h in [3, 8, 11, 17, 20]})
 
