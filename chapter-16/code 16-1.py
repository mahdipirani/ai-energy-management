import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(161)
 
# --- DR season: 14 notified days, summer 1406 (Tir..Shahrivar) ------------
n = 14
tfc = np.sort(rng.normal(40.6, 1.6, n).round(1))      # D-1 14:00 forecast
tact = (tfc + rng.normal(0, 0.9, n)).round(1)         # actual Tmax
 
def p_peak(t):
    return float(np.where(t >= 41, 0.90, np.where(t >= 39, 0.65, 0.35)))
 
# --- baseline: grid import 11:00-17:00 on hot workdays --------------------
# (PV already on; import rises with Tmax via HVAC). Calibrate on 30 days:
t_cal = rng.normal(39.5, 1.9, 30)
imp_cal = 4700 + 65 * (t_cal - 38) + rng.normal(0, 80, 30)
b1, b0 = np.polyfit(t_cal, imp_cal, 1)
print(f"baseline model : import_11_17 = {b1:.1f} x Tmax {b0:+.0f} kWh "
      f"(day-of temp adjustment)")
 
# --- response ladder: kWh moved out of 11-17, and cost in M-IRR -----------
levels = {'A': (650, 3),       # pre-cool + air-schedule
          'B': (3150, 111),    # A + battery midday dispatch (2500 kWh)
          'C': (4650, 369)}    # B + hold one melt batch (1500 kWh)
reward = 130.0                 # k-IRR per verified kWh (2 x peak tariff)
 
rows = []
for i in range(n):
    p = p_peak(tfc[i])
    ev = {k: p * (v[0] * reward / 1000 - v[1]) - (1 - p) * 0.1 * v[1]
          for k, v in levels.items()}
    best = max(ev, key=ev.get)
    declared = rng.random() < p
    base = b0 + b1 * tact[i]
    if declared:
        actual = base - levels[best][0] + rng.normal(0, 60)
    else:
        actual = base + rng.normal(0, 60)             # response stood down
    cost = levels[best][1] if declared else 0.1 * levels[best][1]
    ver = max(0.0, base - actual)
    pay = ver * reward / 1000 if declared else 0.0
    rows.append(dict(day=i + 1, Tfc=tfc[i], Tact=tact[i],
                     P=round(p, 2),
                     level=best if declared else best + ' (stood down)',
                     ver_kwh=round(ver), reward_M=round(pay, 1),
                     cost_M=round(cost, 1), net_M=round(pay - cost, 1)))
 
df = pd.DataFrame(rows)
tot_v = df['ver_kwh'].sum() / 1000
tot_r, tot_c = df['reward_M'].sum() / 1000, df['cost_M'].sum() / 1000
print(df.to_string(index=False))
print(f"season totals  : {tot_v:.1f} MWh verified, reward {tot_r:.2f} "
      f"G-IRR, cost {tot_c:.2f} G-IRR, net {tot_r - tot_c:.2f} G-IRR")
print("level C never wins on pure economics: reward 195 M-IRR < "
      "cost 255 M-IRR per batch (only reliability value closes the gap)")
 
# --- figure ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
 
ax = axes[0]
x = np.arange(1, n + 1)
basel = [b0 + b1 * t for t in tact]
colmap = {'A': '#0E8A6A', 'B': '#0B4E6B', 'C': '#B46C16'}
cols = ['#94A3B8' if 'stood' in l else colmap.get(l[0], '#94A3B8')
         for l in df['level']]
ax.bar(x - 0.2, basel, 0.38, color='#C9D6DF', label='baseline (adj.)')
ax.bar(x + 0.2, [v - k for v, k in zip(basel, df['ver_kwh'])], 0.38,
       color=cols, label='actual import')
for xi, l in zip(x, df['level']):
    if 'stood' in l:
        ax.text(xi, b0 + b1 * tact[xi - 1] + 120, 'x', color='#94A3B8',
                ha='center', fontsize=9)
ax.set_xlabel('notified day (1..14, sorted by forecast Tmax)')
ax.set_ylabel('grid import 11:00-17:00 (kWh)')
ax.set_title('event days: baseline vs actual')
ax.legend(fontsize=8.5)
ax.grid(alpha=0.25, ls=':', axis='y')
 
ax = axes[1]
cum_r = np.cumsum(df['reward_M']) / 1000
cum_c = np.cumsum(df['cost_M']) / 1000
ax.plot(x, cum_r, color='#0E8A6A', lw=2.2, marker='o', ms=3.5,
        label='cumulative reward')
ax.plot(x, cum_c, color='#B46C16', lw=2.2, marker='s', ms=3.5,
        label='cumulative cost')
ax.plot(x, cum_r - cum_c, color='#0A3D53', lw=2.6,
        label='cumulative net')
ax.fill_between(x, 0, cum_r - cum_c, color='#0E8A6A', alpha=0.12)
ax.set_xlabel('notified day')
ax.set_ylabel('G-IRR')
ax.set_title(f'season net: +{tot_r - tot_c:.2f} G-IRR in 14 days')
ax.legend(fontsize=8.5)
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
