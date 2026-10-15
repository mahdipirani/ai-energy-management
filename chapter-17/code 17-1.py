import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(171)
 
# --- benchmarking round of the provincial foundry group -------------------
# 12 peers + PARS, one production year, anonymized, standard report form
name = [f'P{i:02d}' for i in range(1, 13)] + ['PARS']
tons = np.array([7.2, 14.8, 9.6, 21.5, 11.3, 18.4, 8.1, 12.9, 16.2,
                 10.1, 6.8, 13.6, 10.32]) * 1000        # t / year
furnace = np.array(['ind', 'ind', 'EAF', 'ind', 'ind', 'ind', 'EAF',
                    'ind', 'ind', 'EAF', 'ind', 'EAF', 'ind'])
metered = np.array([True] * 13)
metered[5] = False                                     # P06 self-reported
 
sec = 445 + 1_750_000 / tons + 85 * (furnace == 'EAF')
sec = sec + rng.normal(0, 12, 13)
sec[12] = 622.0                                        # PARS, from our meters
sec[5] = 505.0                                         # P06 claim
 
df = pd.DataFrame(dict(plant=name, tons=tons.round(0), SEC=sec.round(0),
                       furnace=furnace, metered=metered))
df['raw_rank'] = df['SEC'].rank().astype(int)
 
# --- normalize: SEC ~ a + b/tons + furnace-type, metered plants only ------
fit = df[df.metered]
X = np.column_stack([np.ones(fit.shape[0]), 1 / fit['tons'],
                     (fit['furnace'] == 'EAF').astype(float)])
beta, *_ = np.linalg.lstsq(X, fit['SEC'], rcond=None)
a, b, c = beta
df['SEC_adj'] = (df['SEC'] - c * (df.furnace == 'EAF')).round(0)
df['adj_rank'] = np.nan
df.loc[df.metered, 'adj_rank'] = (
    df.loc[df.metered, 'SEC_adj'].rank().astype(float))
 
print(df[['plant', 'tons', 'SEC', 'furnace', 'metered',
          'SEC_adj', 'raw_rank', 'adj_rank']].to_string(index=False))
print(f"normalization model : SEC = {a:.0f} + {b / 1e6:.2f}e6/tons "
      f"+ {c:.0f} [EAF]")
 
q1 = df.loc[df.metered, 'SEC_adj'].quantile(0.25)
med = df.loc[df.metered, 'SEC_adj'].median()
gap_kwh = (622 - q1) * 10_320
print(f"adjusted quartiles  : Q1 = {q1:.0f}, median = {med:.0f} kWh/t")
print(f"PARS raw rank       : 7 of 13 -> adjusted rank "
      f"{int(df.loc[12, 'adj_rank'])} of 12 metered")
print(f"gap to Q1           : {622 - q1:.0f} kWh/t = {gap_kwh / 1000:.0f} "
      f"MWh/yr ~ {gap_kwh * 36.1 / 1e6:.1f} G-IRR/yr")
 
# --- figure ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 5.4))
xs = np.linspace(6000, 23000, 200)
ax.plot(xs, a + b / xs, color='#0B4E6B', lw=2, label='fit: induction')
ax.plot(xs, a + b / xs + c, color='#0B4E6B', lw=1.6, ls='--',
        label='fit: EAF (+85 kWh/t)')
for ftype, mark in (('ind', 'o'), ('EAF', 's')):
    m = (df.furnace == ftype) & df.metered
    ax.scatter(df.loc[m, 'tons'], df.loc[m, 'SEC'], s=52, marker=mark,
               color='#1F6480' if ftype == 'ind' else '#B46C16',
               alpha=0.85, edgecolor='white', zorder=3)
ax.scatter([df.loc[12, 'tons']], [622], s=230, marker='*',
           color='#0E8A6A', edgecolor='#0A3D53', lw=1.2, zorder=4,
           label='PARS-Ayandeh (622)')
ax.scatter([18400], [505], s=90, facecolor='none', edgecolor='#B46C16',
           lw=1.8, zorder=4, label='P06 self-reported (excluded)')
ax.axhline(q1, color='#0E8A6A', ls=':', lw=1.8)
ax.text(21500, q1 + 4, f'adj. Q1 = {q1:.0f}', color='#0E8A6A', fontsize=9)
ax.axhline(med, color='#475569', ls=':', lw=1.2)
ax.text(21000, med + 8, f'adj. median = {med:.0f}', color='#475569',
        fontsize=9)
ax.set_xlabel('annual production (t/year)')
ax.set_ylabel('SEC (kWh per ton)')
ax.set_title('benchmarking: raw SEC vs production, furnace type')
ax.legend(fontsize=8.6)
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
