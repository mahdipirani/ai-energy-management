import numpy as np
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(151)
 
# --- 14 months of billing data (the kvarh row of the bill) ---------------
# purchased active energy ~535 MWh/month, production ~860 t/month
m_lab = ['Far05', 'Ord05', 'Kho05', 'Tir05', 'Mor05', 'Sha05', 'Mehr05',
         'Aban05', 'Azar05', 'Dey05', 'Bah05', 'Esf05', 'Far06', 'Ord06']
kwh = rng.normal(535_000, 11_000, 14).round(0)
kvarh = (kwh * np.tan(np.arccos(0.82)) * rng.normal(1.00, 0.015, 14)).round(0)
pf = kwh / np.sqrt(kwh ** 2 + kvarh ** 2)
 
# --- bill model: TOU energy + reactive penalty ----------------------------
# TOU split (ch.3): ~30% night @18, 45% mid @32, 25% peak @65 k-IRR/kWh
# penalty rule (simplified, utility-style): if PF < 0.90:
#     penalty = active_kwh * (0.90 / PF - 1) * blended_rate
rate = 0.30 * 18 + 0.45 * 32 + 0.25 * 65
energy_cost = kwh * rate / 1e6                        # G-IRR
penalty = np.where(pf < 0.90, kwh * (0.90 / pf - 1) * rate, 0.0) / 1e6
 
print(f"blended TOU rate       : {rate:.1f} k-IRR/kWh")
print(f"energy cost / month    : {energy_cost.mean():.2f} G-IRR")
print(f"average power factor   : {pf.mean():.3f}")
print(f"reactive penalty /month: {penalty[:11].mean():.2f} G-IRR "
      f"= {penalty[:11].mean() / energy_cost.mean() * 100:.1f}% of bill")
 
# --- capacitor bank sizing -------------------------------------------------
# size on peak demand (~1010 kW after the night shift), target PF = 0.96
P_peak = 1010.0
tan1, tan2 = np.tan(np.arccos(0.82)), np.tan(np.arccos(0.96))
qc = P_peak * (tan1 - tan2)
bank = 50 * np.ceil(qc / 50)                          # 50-kvar steps
capex = bank * 4.9 / 1000                             # 4.9 M-IRR per kvar
payback = capex / penalty[:11].mean()                 # months
 
print(f"required Qc            : {qc:.0f} kvar")
print(f"installed bank         : {bank:.0f} kvar (9 x 50 kvar, auto)")
print(f"capex                  : {capex:.2f} G-IRR")
print(f"payback                : {payback:.1f} months")
 
# --- commissioning in Esfand 1405: the next two bills ---------------------
# the automatic controller holds PF at 0.95-0.96 -> penalty row = 0
penalty[11:] = 0.0
print("after commissioning    : PF = 0.955 avg on Far06/Ord06 bills, "
      "penalty = 0")
 
# --- figure ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
 
ax = axes[0]
P = 1000.0
for pf_i, color, name in [(0.82, '#B46C16', 'before (PF 0.82)'),
                          (0.96, '#0E8A6A', 'after caps (PF 0.96)')]:
    Q = P * np.tan(np.arccos(pf_i))
    ax.annotate('', xy=(P, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='#0B4E6B', lw=2.6))
    ax.annotate('', xy=(P, Q), xytext=(P, 0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=2.2))
    ax.plot([0, P], [0, Q], color=color, lw=1.5, ls='--', alpha=0.8)
    ax.text(P * 1.02, Q, name, color=color, fontsize=9, va='center')
    ax.text(P * 1.02, Q - 55, f'Q = {Q:.0f} kvar', color=color, fontsize=8.4)
ax.text(P * 0.30, -120, 'P = 1000 kW (active, unchanged)',
        color='#0B4E6B', fontsize=9)
ax.set_xlim(-80, 1620)
ax.set_ylim(-160, 800)
ax.set_xlabel('active power P (kW)')
ax.set_ylabel('reactive power Q (kvar)')
ax.set_title('capacitors cut Q and S, never P')
ax.grid(alpha=0.25, ls=':')
 
ax = axes[1]
x = np.arange(1, 15)
ax.bar(x, penalty, color=['#B46C16'] * 11 + ['#0E8A6A'] * 3, alpha=0.9)
ax.plot(x, np.cumsum(penalty), color='#0A3D53', lw=2.0, marker='o', ms=3.5,
        label='cumulative')
ax.axvline(11.5, color='#475569', ls='--', lw=1.2)
ax.text(7.3, penalty.max() * 1.20, 'caps on (Esfand 05)', fontsize=8.6,
        color='#475569')
ax.set_xticks(x)
ax.set_xticklabels(m_lab, rotation=60, fontsize=7.2)
ax.set_ylabel('reactive penalty (G-IRR/month)')
ax.set_title('the penalty row, before and after')
ax.legend(fontsize=8.5)
ax.grid(alpha=0.25, ls=':', axis='y')
plt.tight_layout()
plt.show()
 
