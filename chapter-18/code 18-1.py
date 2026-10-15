import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(181)
 
# --- 24 months of the EnMS: Far 1405 .. Esf 1406 ---------------------------
m = ['Far', 'Ord', 'Kho', 'Tir', 'Mor', 'Sha', 'Mehr', 'Aban', 'Azar',
     'Dey', 'Bah', 'Esf']
lab = [f'{x}05' for x in m] + [f'{x}06' for x in m]
tons = np.concatenate([rng.normal(858, 22, 12), rng.normal(868, 20, 12)])
 
# --- site EnPI (purchased + on-site solar, per ton) ------------------------
# steps: heater fix (Aban05), air-leak program (Bah05), alert (Mor06)
site = np.array([
    0.611, 0.612, 0.613, 0.614, 0.612, 0.613, 0.613,
    0.599, 0.598, 0.597,
    0.568, 0.566,
    0.566, 0.565, 0.567, 0.569, 0.579, 0.570, 0.568,
    0.567, 0.566, 0.565, 0.566, 0.565]) + rng.normal(0, 0.002, 24)
 
# --- on-site solar (ch.15) cuts PURCHASED energy, not site energy ----------
d = np.array([3.0, 3.7, 4.5, 5.2, 5.9, 6.3, 6.2, 5.8, 5.2, 4.1, 3.1, 2.8])
days = np.array([31, 31, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
pv = np.concatenate([np.zeros(12), d * days * 400.0 / 1000])   # MWh/mo
purchased_enpi = site - pv / tons
 
# --- individual control chart with documented re-baselines -----------------
windows = [(0, 7), (7, 10), (10, 15), (20, 24)]        # baseline windows
alerts = [(7, 'LOW', 'heater fix verified (ch.8)'),
          (10, 'LOW', 'compressed-air leak program'),
          (16, 'HIGH', 'chiller setpoints drifted -> corrected')]
 
print('--- re-baselines (ISO 50006): window, mean, 2-sigma limits ---')
for lo_, hi_ in windows:
    seg = site[lo_:hi_]
    mu, sd = seg.mean(), seg.std(ddof=1)
    print(f"{lab[lo_]:>7s}..{lab[hi_ - 1]:<7s} mu={mu:.4f} "
          f"limits=[{mu - 2 * sd:.4f}, {mu + 2 * sd:.4f}]")
 
for idx, direction, why in alerts:
    print(f"alert {lab[idx]:>7s} {direction}: {why}")
 
# --- verified savings CUSUM by source (G-IRR) ------------------------------
months = np.arange(24)
sav = pd.DataFrame(0.0, index=months, columns=[
    'night-shift arbitrage', 'heater fix', 'air-leak program',
    'PF capacitors', 'rooftop PV', 'battery arbitrage', 'DR season'])
sav.loc[8:23, 'night-shift arbitrage'] = 2.61      # Azar05.. (M&V ch.14)
sav.loc[7:23, 'heater fix'] = 0.42
sav.loc[10:23, 'air-leak program'] = 0.95
sav.loc[11:23, 'PF capacitors'] = 1.88
sav.loc[12:23, 'rooftop PV'] = pv[12:] * 32 / 1000
sav.loc[14:23, 'battery arbitrage'] = 2.64          # Khordad06..
sav.loc[17, 'DR season'] = 3.57                     # Shahrivar06 close
cum = sav.cumsum()
total = cum.iloc[-1].sum()
 
print('--- cumulative verified savings by source (G-IRR) ---')
for col in cum.columns:
    print(f"{col:<24s}: {cum[col].iloc[-1]:6.1f}")
print(f"{'TOTAL (24 months)':<24s}: {total:6.1f} G-IRR")
run_rate = 2.61 + 0.42 + 0.95 + 1.88 + pv[23] * 32 / 1000 + 2.64
print(f"run-rate at Esf06 (no new projects): {run_rate:.1f} G-IRR/month"
      f" -> {run_rate * 12:.0f} G-IRR/year")
 
# --- figure ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
 
ax = axes[0]
ax.plot(months, site, color='#0B4E6B', lw=2.2, marker='o', ms=3.5,
        label='site EnPI (purch + PV)')
ax.plot(months, purchased_enpi, color='#0E8A6A', lw=1.6, ls='--',
        label='purchased EnPI')
for lo_, hi_ in windows:
    seg = site[lo_:hi_]
    mu, sd = seg.mean(), seg.std(ddof=1)
    ax.fill_between([lo_, hi_], mu - 2 * sd, mu + 2 * sd,
                    color='#0B4E6B', alpha=0.10)
    ax.hlines(mu, lo_, hi_ + (3 if hi_ < 24 else 0), color='#0B4E6B',
              lw=1.1, alpha=0.7)
for idx, direction, why in alerts:
    ax.scatter([idx], [site[idx]], s=130, facecolor='none',
               edgecolor='#B46C16' if direction == 'HIGH' else '#0E8A6A',
               lw=2, zorder=5)
ax.annotate('heater fix', (7, site[7]), (2.2, 0.575), fontsize=8.4,
            arrowprops=dict(arrowstyle='->', color='#475569', lw=1))
ax.annotate('leak program', (10, site[10]), (4.5, 0.545), fontsize=8.4,
            arrowprops=dict(arrowstyle='->', color='#475569', lw=1))
ax.annotate('PV starts', (12, purchased_enpi[12]), (13.2, 0.636),
            fontsize=8.4, arrowprops=dict(arrowstyle='->',
                                          color='#475569', lw=1))
ax.annotate('alert: HIGH', (16, site[16]), (15.2, 0.61), fontsize=8.4,
            arrowprops=dict(arrowstyle='->', color='#B46C16', lw=1))
ax.set_xticks(months[::2])
ax.set_xticklabels([lab[i] for i in range(0, 24, 2)], rotation=60,
                   fontsize=6.8)
ax.set_ylabel('EnPI (MWh purchased+PV / ton)')
ax.set_title('EnPI control chart, three re-baselines')
ax.legend(fontsize=8.4, loc='lower left')
ax.grid(alpha=0.25, ls=':')
 
ax = axes[1]
colors = ['#0B4E6B', '#0E8A6A', '#1F6480', '#B46C16', '#7FB069',
          '#0A3D53', '#C98A2B']
ax.stackplot(months, [cum[c] for c in cum.columns], labels=cum.columns,
             colors=colors, alpha=0.9)
ax.set_xticks(months[::2])
ax.set_xticklabels([lab[i] for i in range(0, 24, 2)], rotation=60,
                   fontsize=6.8)
ax.set_ylabel('cumulative verified savings (G-IRR)')
ax.set_title(f'two years, {total:.0f} G-IRR — every source on one chart')
ax.legend(fontsize=7.4, loc='upper left')
ax.grid(alpha=0.25, ls=':')
plt.tight_layout()
plt.show()
 
