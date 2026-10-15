import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# --- 24 months, seven sources — same ledger as ch.18 ----------------
m = ['Far', 'Ord', 'Kho', 'Tir', 'Mor', 'Sha', 'Mehr', 'Aban',
     'Azar', 'Dey', 'Bah', 'Esf']
lab = [f'{x}05' for x in m] + [f'{x}06' for x in m]
months = np.arange(24)
 
sav = pd.DataFrame(0.0, index=months, columns=[
    'night-shift', 'heater', 'air-leak', 'PF-capacitors',
    'rooftop-PV', 'battery', 'DR-season'])
sav.loc[8:23, 'night-shift'] = 2.61     # G-IRR per month, Azar05..
sav.loc[7:23, 'heater'] = 0.42
sav.loc[10:23, 'air-leak'] = 0.95
sav.loc[11:23, 'PF-capacitors'] = 1.88
d = np.array([3.0, 3.7, 4.5, 5.2, 5.9, 6.3, 6.2, 5.8, 5.2, 4.1,
              3.1, 2.8])
days = np.array([31, 31, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
pv = d * days * 400.0 / 1000            # MWh per month (400 kWp)
sav.loc[12:23, 'rooftop-PV'] = pv * 32 / 1000
sav.loc[14:23, 'battery'] = 2.64        # Khordad06..
sav.loc[17, 'DR-season'] = 3.57         # Shahrivar06 settlement
 
cum = sav.cumsum()
total = cum.iloc[-1].sum()
capex = 0.4 + 2.2 + 68.0 + 60.0         # timer, capacitors, PV, battery
run_rate = (2.61 + 0.42 + 0.95 + 1.88
            + pv[-1] * 32 / 1000 + 2.64)
proj07 = run_rate * 12
 
print('=== Pars-Ayandeh — annual report, Far05..Esf06 ===')
for c in cum.columns:
    state = 'seasonal' if c == 'DR-season' else 'active'
    print(f'{c:<14s} {cum[c].iloc[-1]:7.1f}  {state}')
print(f"{'TOTAL':<14s} {total:7.1f}  G-IRR")
print(f"capex deployed: {capex:5.1f} G   "
      f"return / capex: {total / capex:.2f} x")
print(f"run-rate Esf06: {run_rate:5.1f} G/month "
      f"-> projection 1407: {proj07:5.0f} G")
 
# --- figure: monthly bars, cumulative, 1407 projection --------------
fig, ax = plt.subplots(figsize=(10.4, 4.6))
per_month = sav.sum(axis=1)
ax.bar(months, per_month, color='#0E8A6A', alpha=0.8,
       label='verified savings / month')
ax.plot(months, cum.sum(axis=1), color='#0B4E6B', lw=2.4,
        marker='o', ms=3.5, label='cumulative')
proj_x = np.arange(24, 37)
ax.plot(proj_x, total + run_rate * np.arange(1, 14),
        color='#B46C16', ls='--', lw=2.0,
        label='1407 projection (run-rate)')
ax.axvline(23.5, color='#475569', ls=':', lw=1)
ax.text(0.8, total * 0.86, f'total: {total:.1f} G', fontsize=10.5,
        color='#0B4E6B', fontweight='bold')
ax.text(25.2, total + 6, f'{proj07:.0f} G', fontsize=10.5,
        color='#B46C16', fontweight='bold')
ax.set_xticks(np.arange(0, 37, 2))
ax.set_xticklabels(lab[::2] + ['Far07', 'Kho07', 'Mor07', 'Mehr07',
                               'Azar07', 'Bah07', 'Far08'],
                   rotation=60, fontsize=6.4)
ax.set_ylabel('G-IRR')
ax.set_title('two years verified, one year projected')
ax.legend(fontsize=8.6, loc='upper left')
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
