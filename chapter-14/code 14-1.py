import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# --- remaining purchasable value in the peak window, after the program --
T_OFF, T_PEAK, T_MID = 18.0, 65.0, 32.0
ETA, P_MAX, DAYS = 0.88, 500.0, 312          # round-trip, kW, days/yr
CAPEX_FIX, CAPEX_KWH = 6000.0, 27.0          # M-IRR (fixed / per kWh)
LIFE, DISC, DEGR = 10, 0.25, 0.02
 
# winter day after ch.13 program: peak-window fixed load
PEAK_LEFT = 4438 - 1950 - 204                # kWh per day
 
value_kwh = T_PEAK - T_OFF / ETA             # k-IRR per discharged kWh
print(f"value of each discharged kWh : {value_kwh:.1f} k-IRR "
      f"(sell at {T_PEAK:.0f}, buy at {T_OFF / ETA:.1f})")
print(f"peak-window energy left      : {PEAK_LEFT / 1000:.2f} MWh/day")
print(f"battery bigger than that     : buys nothing")
 
def npv(irr, flows):
    return sum(f / (1 + irr) ** t for t, f in enumerate(flows))
 
def irr_bisect(flows):
    lo, hi = -0.9, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid, flows) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
 
def economics(E):                            # E in kWh
    dis = min(ETA * E, PEAK_LEFT, P_MAX * 6)  # kWh per day
    sav1 = dis * value_kwh * DAYS / 1000      # M-IRR per year
    capex = CAPEX_FIX + CAPEX_KWH * E
    flows = [-capex] + [sav1 * (1 - DEGR) ** t for t in range(LIFE)]
    return dis, sav1, capex, capex / sav1, npv(DISC, flows), irr_bisect(flows)
 
sizes = [500, 1000, 1500, 2000, 2600, 3000, 3500, 4000]
rows = []
for E in sizes:
    dis, sav1, capex, pb, nv, irr = economics(E)
    rows.append([E, round(dis), round(sav1), round(capex),
                 round(pb, 1), round(nv), round(100 * irr)])
df = pd.DataFrame(rows, columns=['size kWh', 'discharge kWh/d',
                                 'saving M-IRR/yr', 'capex M-IRR',
                                 'payback yr', 'NPV@25% M-IRR', 'IRR %'])
print(df.to_string(index=False))
print('---')
knee = PEAK_LEFT / ETA
print(f"knee of the curve: {knee:.1f} kWh "
      f"(0.88 x E = {PEAK_LEFT} kWh left to serve)")
 
# --- what if the program had never run? --------------------------------
PEAK_LEFT0 = 4438.0                          # today, no night-shift
knee0 = PEAK_LEFT0 / ETA
d0, s0, cx0, pb0, nv0, irr0 = economics(knee0)
print(f"without the program: optimal battery {knee0:.0f} kWh, "
      f"capex {cx0:.0f} M-IRR")
print(f"with the program   : optimal battery {knee:.0f} kWh, "
      f"capex {economics(knee)[2]:.0f} M-IRR")
 
# --- tariff sensitivity at the knee ------------------------------------
print('--- sensitivity of value per kWh (peak tariff) ---')
capex_knee = economics(knee)[2]
for tp in (50.0, 65.0, 80.0):
    v = tp - T_OFF / ETA
    print(f"peak {tp:.0f} M-IRR/MWh -> value {v:5.1f} k-IRR/kWh, "
          f"payback at knee {capex_knee / (PEAK_LEFT * v * DAYS / 1000):.1f} yr")
 
# --- chart: sizing curve ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
Es = np.linspace(300, 4500, 60)
dis = [min(ETA * E, PEAK_LEFT) / 1000 for E in Es]
sav = [d * 1000 * value_kwh * DAYS / 1000 / 1000 for d in dis]
axes[0].plot(Es / 1000, dis, color='#0E8A6A', lw=2.2)
axes[0].axvline(knee / 1000, color='#B46C16', ls='--', lw=1.4)
axes[0].text(knee / 1000 + 0.1, 1.1, 'knee 2.6', color='#B46C16',
             fontsize=9)
axes[0].set_xlabel('battery size (MWh)')
axes[0].set_ylabel('discharged energy (MWh/day)')
axes[0].set_title('Buyer of a shrinking commodity')
axes[0].grid(alpha=0.25, ls=':')
nv = [economics(E)[4] / 1000 for E in Es]
axes[1].plot(Es / 1000, nv, color='#0B4E6B', lw=2.2)
axes[1].axvline(knee / 1000, color='#B46C16', ls='--', lw=1.4)
axes[1].axhline(0, color='#8B2E2E', lw=1, ls=':')
axes[1].set_xlabel('battery size (MWh)')
axes[1].set_ylabel('NPV at 25% (G-IRR)')
axes[1].set_title('The optimum sits at the knee')
axes[1].grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
