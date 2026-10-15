import numpy as np
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(152)
 
# --- rooftop PV on warehouse no.2 : 400 kWp --------------------------------
# 3,400 m2 of the 6,000 m2 roof (8.5 m2/kWp), 25-deg tilt, PR ~0.80
kwp = 400.0
m_lab = ['Far', 'Ord', 'Kho', 'Tir', 'Mor', 'Sha', 'Mehr', 'Aban',
         'Azar', 'Dey', 'Bah', 'Esf']
d = np.array([3.0, 3.7, 4.5, 5.2, 5.9, 6.3, 6.2, 5.8, 5.2, 4.1, 3.1, 2.8])
days = np.array([31, 31, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
m_kwh = d * days * kwp                                # kWh per month
annual = m_kwh.sum()
print(f"annual production   : {annual / 1000:.0f} MWh "
      f"({annual / kwp:.0f} kWh per kWp)")
 
# --- typical-day load, post night-shift (ch.13 dispatch) ------------------
def load_kw(h, friday=False):
    if friday:
        return 410 + 25 * np.sin((h - 6) * np.pi / 12) ** 2
    if h < 7:
        return 770
    if h < 16:
        return 1120
    if h < 17:
        return 650
    if h < 23:
        return 450
    return 640
 
def daylen(m):
    return 12 + 2.3 * np.cos(2 * np.pi * (m - 5.75) / 12)
 
def pv_kw(h, m):
    dl = daylen(m)
    rise, sunset = 12 - dl / 2, 12 + dl / 2
    if h <= rise or h >= sunset:
        return 0.0
    pmax = d[m - 1] * kwp * np.pi / (2 * dl)
    return pmax * np.sin(np.pi * (h - rise) / dl)
 
# self-consumption check: worst case = max-PV day vs Friday (lowest) load
hours = np.arange(0, 24, 0.25)
pv_tir = np.array([pv_kw(h, 7) for h in hours])
load_fri = np.array([load_kw(h, True) for h in hours])
export = np.clip(pv_tir - load_fri, 0, None).sum() * 0.25
print(f"worst-case export   : {export:.1f} kWh/day (PV peak "
      f"{pv_tir.max():.0f} kW vs min Friday load {load_fri.min():.0f} kW)"
      f" -> 100% self-consumed, zero export contract")
 
# --- economics: the language of chapter 14 ---------------------------------
capex = 170 * kwp / 1000                              # G-IRR (170 M-IRR/kWp)
om = 0.010 * capex                                    # G-IRR / year
disc, deg = 0.25, 0.005
val = annual / 1000 * 32 / 1000                       # G-IRR/yr, all at mid 32
 
def npv_at(r):
    return -capex + sum((val * (1 - deg) ** (y - 1) - om) / (1 + r) ** y
                        for y in range(1, 21))
 
npv = npv_at(disc)
lo, hi = 0.0, 1.0
for _ in range(60):
    mid = (lo + hi) / 2
    if npv_at(mid) > 0:
        lo = mid
    else:
        hi = mid
irr = (lo + hi) / 2
 
cum, pay = -capex, None
for y in range(1, 21):
    cash = val * (1 - deg) ** (y - 1) - om
    if cum + cash >= 0 and pay is None:
        pay = y - 1 + (-cum / cash)
    cum += cash
 
print(f"capex               : {capex:.1f} G-IRR")
print(f"net year-1 cash     : {val - om:.1f} G-IRR (every kWh at mid 32)")
print(f"payback             : {pay:.1f} years")
print(f"NPV @ 25% / 20 yr   : {npv:.1f} G-IRR")
print(f"IRR                 : {irr * 100:.0f}% (hurdle 25%)")
 
# --- figure ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
 
ax = axes[0]
hw = np.arange(0, 24, 0.25)
lw = np.array([load_kw(h) for h in hw])
pw = np.array([pv_kw(h, 7) for h in hw])
ax.plot(hw, lw, color='#0B4E6B', lw=2.2, label='site load (workday)')
ax.fill_between(hw, 0, pw, color='#0E8A6A', alpha=0.55,
                label='PV 400 kWp (self-consumed)')
ax.plot(hw, pw, color='#0E8A6A', lw=1.8)
ax.axvspan(17, 23, color='#B46C16', alpha=0.08)
ax.text(17.00, 1000, 'peak tariff window', fontsize=8,
        color='#B46C16')
ax.set_xlabel('hour of day (Tir, typical workday)')
ax.set_ylabel('power (kW)')
ax.set_title('the sun works midday, the bill peaks in the evening')
ax.set_xticks(range(0, 25, 2))
ax.legend(fontsize=8.5, loc='lower right')
ax.grid(alpha=0.25, ls=':')
 
ax = axes[1]
x = np.arange(1, 13)
share = m_kwh / 1000 / 535 * 100
ax.bar(x, m_kwh / 1000, color='#0E8A6A', alpha=0.8,
       label='PV production (MWh)')
ax2 = ax.twinx()
ax2.plot(x, share, color='#B46C16', lw=2, marker='o', ms=4,
         label='share of purchase (%)')
ax2.set_ylim(0, 20)
ax.set_xticks(x)
ax.set_xticklabels(m_lab, fontsize=8.6)
ax.set_ylabel('MWh / month')
ax2.set_ylabel('% of purchased energy', color='#B46C16')
ax.set_title(f'annual {annual / 1000:.0f} MWh = '
             f'{share.mean():.1f}% of purchase (avg)')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc='upper left')
ax.grid(alpha=0.25, ls=':', axis='y')
plt.tight_layout()
plt.show()
 
