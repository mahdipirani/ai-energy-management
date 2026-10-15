import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# --- typical autumn workday, hourly site load (kW), matches ch. 7 ------
base = np.array([540, 540, 540, 540, 540, 540, 540, 850,
                 1120, 1120, 1120, 1120, 1120, 1120, 1120, 1120,
                 1120, 900, 700, 700, 700, 700, 560, 560])
hours = np.arange(24)
 
# --- TOU periods and tariffs (same as ch. 3) ---------------------------
def period(h):
    if h < 7:  return 'off'
    if 17 <= h < 23: return 'peak'
    return 'mid'
tariff = np.array([18.0 if period(h) == 'off' else
                   65.0 if period(h) == 'peak' else 32.0
                   for h in hours])
 
# --- today's position of the flexible loads ----------------------------
furnace_now = np.zeros(24); furnace_now[17:23] = 250  # melt #2, 17-22
air_now = np.zeros(24);     air_now[18:23] = 90       # air charge 18-22
fixed = base - furnace_now - air_now
 
def cost(load):        # M-IRR per day
    return (load * tariff).sum() / 1000
 
# --- shift scenarios ----------------------------------------------------
air_night = np.zeros(24); air_night[0:5] = 90          # to 00-05
f_even_half = np.zeros(24); f_even_half[17:23] = 125  # 750 kWh stays 17-22
f_night_half = np.zeros(24); f_night_half[0:6] = 125  # 750 kWh to night
f_full = np.zeros(24); f_full[0:6] = 250              # 1500 kWh to night
 
S = {}
S['S0 today'] = fixed + furnace_now + air_now
S['S1 air only'] = fixed + furnace_now + air_night
S['S2 air + half melt'] = (fixed + f_even_half + f_night_half
                           + air_night)
S['S3 full program'] = fixed + f_full + air_night
 
rows = []
for name, load in S.items():
    peakw = load[(hours >= 17) & (hours < 23)].sum()
    rows.append([name, round(cost(load), 1), round(load.max()),
                 round(load[:7].max()),
                 round(peakw / 1000, 2), round(load.mean() / load.max(), 2)])
df = pd.DataFrame(rows, columns=['scenario', 'cost M-IRR/day',
                                 'site peak kW', 'night peak kW',
                                 'peak-window MWh', 'load factor'])
print(df.to_string(index=False))
print('---')
for name in list(S)[1:]:
    save = cost(S['S0 today']) - cost(S[name])
    print(f"{name:20s}: saving {save:5.1f} M-IRR/day "
          f"(~{26 * save / 1000:4.2f} G-IRR/month)")
 
# --- chart: today vs S3 over the tariff bands ---------------------------
fig, ax = plt.subplots(figsize=(10, 4.3))
ax.plot(hours, S['S0 today'], color='#0B4E6B', lw=2.2, marker='o',
        ms=4, label='today')
ax.plot(hours, S['S3 full program'], color='#0E8A6A', lw=2, ls='--',
        marker='s', ms=3.5, label='S3 night-shift program')
ax.axvspan(0, 7, color='#0E8A6A', alpha=0.07)
ax.axvspan(17, 23, color='#B46C16', alpha=0.10)
ax.text(2.1, 1190, 'off-peak 18', fontsize=8.5, color='#0E8A6A')
ax.text(18.1, 1190, 'peak 65', fontsize=8.5, color='#B46C16')
ax.set_xlabel('hour of day')
ax.set_ylabel('load (kW)')
ax.set_title('Load shifting: the night-shift program (S3)')
ax.legend(fontsize=9)
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
