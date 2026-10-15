import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt
 
# --- winter peak-day forecast (simplified from ch. 11) -----------------
win = np.array([614, 614, 614, 614, 614, 614, 614, 820,
                1140, 1140, 1140, 1166, 1166, 1166, 1166, 1140,
                1140, 948, 720, 720, 720, 720, 610, 600])
furnace_now = np.zeros(24); furnace_now[17:23] = 250
air_now = np.zeros(24);     air_now[18:23] = 90
fixed = win - furnace_now - air_now
 
def period(h):
    if h < 7: return 'off'
    if 17 <= h < 23: return 'peak'
    return 'mid'
tariff = np.array([18.0 if period(h) == 'off' else
                   65.0 if period(h) == 'peak' else 32.0
                   for h in range(24)])
CONTRACT, FEEDER = 1350.0, 300.0
FURN_E, AIR_E = 1500.0, 450.0
 
# --- naive S3 from ch. 12: does it fit the shared feeder? --------------
naive_night = fixed.copy()
naive_night[0:6] += 250
naive_night[0:5] += 90
worst = (naive_night[0:7][naive_night[0:7] > 0]).max()
print(f"naive S3 night load : {naive_night[0:7].max():.0f} kW, "
      f"furnace+air = {250 + 90} kW vs feeder {FEEDER:.0f} kW"
      f"  -->  {'VIOLATION' if 250 + 90 > FEEDER else 'ok'}")
 
# --- LP: variables f[24], a[24], ph ------------------------------------
NF, NA, PH = 24, 24, 1
N = NF + NA + PH
WINDOW = [h for h in list(range(24)) if h >= 22 or h < 7]   # 22-07
bounds = []
for h in range(24):
    bounds.append((0, 250.0 if h in WINDOW else 0))          # furnace
for h in range(24):
    bounds.append((0, 90.0 if h in WINDOW else 0))           # air
bounds.append((0, 240.0))                                    # pre-heat
 
c = np.zeros(N)
for h in range(24):
    c[h] = tariff[h] / 1000                                  # furnace
    c[NF + h] = tariff[h] / 1000                             # air
c[N - 1] = (2 * 0.5 * 18.0 - 3 * (0.85 / 3) * 65.0) / 1000  # pre-heat
 
A_eq, b_eq = [], []
row = np.zeros(N); row[:NF] = 1; A_eq.append(row); b_eq.append(FURN_E)
row = np.zeros(N); row[NF:NF + NA] = 1; A_eq.append(row); b_eq.append(AIR_E)
 
A_ub, b_ub = [], []
for h in WINDOW:                       # shared feeder E-07
    row = np.zeros(N)
    row[h] = 1; row[NF + h] = 1
    A_ub.append(row); b_ub.append(FEEDER)
for h in range(24):                    # contract cap, every hour
    row = np.zeros(N)
    row[h] = 1; row[NF + h] = 1
    if h in (5, 6):
        row[N - 1] = 0.5               # pre-heat charging
    if h in (17, 18, 19):
        row[N - 1] = -0.85 / 3         # pre-heat discharging
    A_ub.append(row); b_ub.append(CONTRACT - fixed[h])
 
res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
              A_eq=np.array(A_eq), b_eq=np.array(b_eq),
              bounds=bounds, method='highs')
x = res.x
f_opt, a_opt, ph = x[:NF], x[NF:NF + NA], x[-1]
load_opt = fixed + f_opt + a_opt
load_opt[17:20] -= 0.85 * ph / 3
load_opt[5:7] += ph / 2
 
def cost(load):
    return (load * tariff).sum() / 1000
 
# --- honest manual fix (what the shift team did after the trip) --------
manual = fixed.copy()
manual[0:6] += 250                      # melt #2, six hard hours
manual[6] += 90                         # air: leftover night hour
manual[22:24] += 90                     # air: two evening hours
manual[0:6] += 30                       # air: feeder scraps 00-05
 
print(f"cost today            : {cost(win):7.1f} M-IRR")
print(f"cost naive S3         : infeasible (feeder E-07 = 340 > 300 kW)")
print(f"cost manual fix       : {cost(manual):7.1f} M-IRR")
print(f"cost LP optimum       : {cost(load_opt):7.1f} M-IRR  "
      f"(pre-heat {ph:.0f} kWh)")
print(f"LP vs manual fix      : {cost(manual) - cost(load_opt):.1f} "
      f"M-IRR/day, {26 * (cost(manual) - cost(load_opt)) / 1000:.2f} "
      f"G-IRR/month")
 
# --- the optimal night schedule ----------------------------------------
night = [(h, f_opt[h], a_opt[h]) for h in WINDOW
         if f_opt[h] + a_opt[h] > 1]
print('--- optimal night schedule (kW) ---')
for h, fv, av in night:
    print(f"hour {h:02d}  furnace {fv:5.1f}  air {av:5.1f}  "
          f"feeder {fv + av:5.1f} / {FEEDER:.0f}")
 
# --- shadow prices ------------------------------------------------------
eq_duals = res.eqlin.marginals
ub_duals = res.ineqlin.marginals
print(f"shadow price of melt energy  : {1000 * eq_duals[0]:.0f} k-IRR/kWh")
print(f"shadow price of air energy   : {1000 * eq_duals[1]:.0f} k-IRR/kWh")
print(f"shadow price of feeder E-07  : {1000 * max(ub_duals[:9]):.0f} "
      f"k-IRR/kW-day (binding: {(np.array(ub_duals[:9]) != 0).sum()} of 9)")
print(f"shadow price of contract     : {1000 * max(ub_duals[9:]):.0f} "
      f"k-IRR/kW-day")
 
# --- growth stress test: does the cap start to bind? --------------------
res2 = linprog(c, A_ub=np.array(A_ub),
               b_ub=np.array(b_ub) - np.concatenate(
                   [np.zeros(9), 1.20 * fixed - fixed]),
               A_eq=np.array(A_eq), b_eq=np.array(b_eq),
               bounds=bounds, method='highs')
print(f"growth +20% stress test      : "
      f"{'INFEASIBLE - capacity alarm' if res2.status == 2 else 'feasible'}")
 
# --- chart: schedules and site load -------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
h24 = np.arange(24)
w = 0.28
axes[0].bar(h24 - w,  f_opt, width=w, color='#0B4E6B', label='furnace (optimal)')
axes[0].bar(h24,      a_opt, width=w, color='#0E8A6A', label='air (optimal)')
axes[0].bar([5 + w, 6 + w], [ph/2, ph/2], width=w,
            color='#B46C16', label='pre-heat')
axes[0].set_ylabel('kW')
axes[0].set_title('The optimal night: soft 7-hour melt + continuous '
                  'air + pre-heat')
axes[0].legend(fontsize=9, ncol=3)
axes[0].grid(alpha=0.25, ls=':')
axes[1].plot(h24, win, color='#0B4E6B', lw=2.2, marker='o', ms=4,
             label='today')
axes[1].plot(h24, load_opt, color='#0E8A6A', lw=2, ls='--',
             label='LP optimum')
axes[1].axhline(CONTRACT, color='#8B2E2E', lw=1.2, ls=':')
axes[1].text(0.3, 1290, 'contract 1350 kW', fontsize=8.5, color='#8B2E2E')
axes[1].set_xlabel('hour of day')
axes[1].set_ylabel('site load (kW)')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
