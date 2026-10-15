import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(1402)
 
# --- pre-program workdays: Far..Aban 1405 (peak-window energy) ---------
months = [('Far', 22, 1.5), ('Ord', 22, 2.5), ('Kho', 21, 2.0),
          ('Tir', 22, 0.0), ('Mor', 23, 0.0), ('Sha', 22, 0.4),
          ('Mehr', 22, 2.0), ('Aban', 21, 3.5)]
days = []
for name, n, hdd in months:
    for _ in range(n):
        h = max(0.0, hdd + rng.normal(0, 0.8))
        pk = 4260 + 30 * h + rng.normal(0, 70)     # kWh in 17-23
        days.append((h, pk, 0))
 
# --- post-program workdays: Azar + Dey (night-shift ON) ----------------
for name, n, hdd in [('Azar', 22, 5.5), ('Dey', 21, 7.0)]:
    for _ in range(n):
        h = max(0.0, hdd + rng.normal(0, 0.8))
        pk = 4260 - 2154 + 30 * h + rng.normal(0, 70)
        days.append((h, pk, 1))
 
df = pd.DataFrame(days, columns=['hdd', 'peak_kwh', 'post'])
pre, post = df[df.post == 0], df[df.post == 1]
 
# --- adjusted baseline: peak-window energy vs weather ------------------
reg = LinearRegression().fit(pre[['hdd']], pre['peak_kwh'])
a, b = reg.intercept_, reg.coef_[0]
r2 = reg.score(pre[['hdd']], pre['peak_kwh'])
print(f"baseline model : peak_kwh = {a:.0f} + {b:.1f} x HDD"
      f"   (R2 = {r2:.2f}, n = {len(pre)})")
 
# --- counterfactual vs actual in the post period -----------------------
post = post.copy()
post['predicted'] = a + b * post['hdd']
post['saved'] = post['predicted'] - post['peak_kwh']
mean, se = post['saved'].mean(), post['saved'].std(ddof=1)
ci = 1.64 * se / np.sqrt(len(post))
print(f"verified shift : {mean:.0f} +- {ci:.0f} kWh/workday (90%)")
print(f"naive, no weather adjustment would have said: "
      f"{pre['peak_kwh'].mean() - post['peak_kwh'].mean():.0f} kWh")
 
tot_mwh = post['saved'].sum() / 1000
sav_girr = post['saved'].sum() * (1950 / 2154 * 47 + 204 / 2154 * 44.0) / 1e6
print(f"total in 43 workdays : {tot_mwh:.1f} MWh moved, "
      f"~{sav_girr:.2f} G-IRR")
print(f"monthly run-rate     : {26 * mean * (1950 / 2154 * 47 + 204 / 2154 * 44.0) / 1e6:.2f} G-IRR")
 
# --- CUSUM of verified savings -----------------------------------------
post = post.reset_index(drop=True)
cusum = post['saved'].cumsum()
 
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
axes[0].scatter(pre['hdd'], pre['peak_kwh'], s=14, color='#0B4E6B',
                alpha=0.75, label='pre (measured)')
axes[0].scatter(post['hdd'], post['peak_kwh'], s=14, color='#B46C16',
                alpha=0.8, label='post (measured)')
xs = np.linspace(0, 9, 50)
axes[0].plot(xs, a + b * xs, color='#0E8A6A', lw=1.8,
             label='adjusted baseline')
axes[0].set_xlabel('heating degree-days (HDD)')
axes[0].set_ylabel('peak-window energy (kWh)')
axes[0].set_title('M&V: baseline, counterfactual, actual')
axes[0].legend(fontsize=8.5)
axes[0].grid(alpha=0.25, ls=':')
axes[1].plot(np.arange(1, len(cusum) + 1), cusum / 1000,
             color='#0E8A6A', lw=2.2)
axes[1].fill_between(np.arange(1, len(cusum) + 1),
                     (cusum - 1.64 * se * np.sqrt(np.arange(1, len(cusum) + 1))) / 1000,
                     (cusum + 1.64 * se * np.sqrt(np.arange(1, len(cusum) + 1))) / 1000,
                     color='#0E8A6A', alpha=0.15, label='90% band')
axes[1].set_xlabel('workday since program start')
axes[1].set_ylabel('cumulative verified shift (MWh)')
axes[1].set_title('CUSUM of verified savings')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
