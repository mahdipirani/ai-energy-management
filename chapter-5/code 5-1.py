import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(1405)
 
# --- synthetic furnace load, 15-min resolution, 7 days -----------------
idx = pd.date_range('2026-09-14 00:00', periods=672, freq='15min')
kw = []
for ts in idx:
    h = ts.hour + ts.minute / 60
    if ts.weekday() == 4:                       # Friday idle
        kw.append(160 + rng.normal(0, 8))
    elif 6 <= h < 14:    kw.append(620 + rng.normal(0, 30))
    elif 14 <= h < 22:   kw.append(480 + rng.normal(0, 20))
    else:                kw.append(175 + rng.normal(0, 10))
raw = pd.Series(np.round(kw, 1), index=idx, name='kw')
 
# --- inject realistic data diseases -------------------------------------
raw.iloc[[210, 421, 640]] = [1180.0, 1250.0, 990.0]   # spikes
raw.iloc[550] = 9999.0                                # sensor glitch
raw.iloc[300] = -50.0                                 # negative glitch
raw.iloc[380:392] = 505.0                             # stuck 3 hours
gap_idx = [96, 97, 200, 338, 339, 470, 471, 472, 590, 620, 30, 31]
raw.iloc[gap_idx] = np.nan                            # comm outage
 
n_missing = int(raw.isna().sum())
print(f"slots                    : {len(raw)}")
print(f"missing at arrival       : {n_missing} "
      f"({100*n_missing/len(raw):.1f} %)")
 
# --- cleaning pipeline ---------------------------------------------------
clean = raw.copy()
 
# rule 1: physical limits of the furnace meter (0 .. 1200 kW)
r1 = (clean < 0) | (clean > 1200)
clean[r1] = np.nan
 
# rule 2: spike — impossible jump from BOTH neighbours
d_prev = (clean - clean.shift()).abs()
d_next = (clean - clean.shift(-1)).abs()
r2 = (d_prev > 250) & (d_next > 250)
clean[r2] = np.nan
 
# rule 3: stuck sensor — run of 8+ identical readings (2 hours)
run_id = (~clean.eq(clean.shift())).cumsum()
run_len = clean.groupby(run_id).transform('size')
r3 = run_len >= 8
clean[r3] = np.nan
 
# rule 4: rebuild the regular grid, interpolate (max 1 hour)
n_before = int(clean.isna().sum())
clean = (clean.asfreq('15min')
              .interpolate(method='time', limit=4,
                           limit_direction='both'))
n_after = int(clean.isna().sum())
 
# --- quality report ------------------------------------------------------
report = pd.Series({
    'raw slots': len(raw),
    'missing at arrival': n_missing,
    'out-of-range removed': int(r1.sum()),
    'spikes removed': int(r2.sum()),
    'stuck slots removed': int(r3.sum()),
    'slots interpolated': n_before - n_after,
    'missing after cleaning': n_after,
    'mean before (kW)': round(raw.mean(), 1),
    'mean after  (kW)': round(clean.mean(), 1),
})
print(report.to_string())
 
# --- before / after figure ------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
axes[0].plot(raw.index, raw, color='#8B2E2E', lw=0.9)
axes[0].set_ylabel('kW (raw)')
axes[0].set_title('Raw: gaps, spikes, glitch 9999 and stuck sensor')
axes[1].plot(clean.index, clean, color='#0E8A6A', lw=0.9)
axes[1].set_ylabel('kW (clean)')
axes[1].set_title('Clean: interpolated (max 1h), stuck gap left open')
for ax in axes:
    ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
