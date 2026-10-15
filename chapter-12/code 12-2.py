import numpy as np
import pandas as pd
 
# --- winter peak-day forecast (simplified from ch. 11, Azar-1) ---------
win = np.array([614, 614, 614, 614, 614, 614, 614, 820,
                1140, 1140, 1140, 1166, 1166, 1166, 1166, 1140,
                1140, 948, 720, 720, 720, 720, 610, 600])
CONTRACT, GROWTH = 1350.0, 1.15
 
print(f"winter day      : {win.sum() / 1000:.1f} MWh, "
      f"peak {win.max():.0f} kW at hour {int(win.argmax())}")
grown = win * GROWTH
print(f"next year +15%  : peak {grown.max():.0f} kW "
      f"vs contract {CONTRACT:.0f} kW")
 
# --- measure 1: two-wave lunch, 11-15, with early-shift make-up --------
m1 = grown.copy()
m1[11:15] -= 70            # staggered lunch waves flatten 11-15
m1[6] += 140               # production made up in the early shift
m1[7] += 140               # (cheap hours, plant under contract)
 
# --- measure 2: office/server cooling pre-run in the cool morning ------
m2 = m1.copy()
m2[8:17] -= 40             # 40 kW of HVAC out of the whole plateau
m2[5] += 180               # same energy, pre-run 05:00-07:00
m2[6] += 180
 
rows = [['today (winter)', win, CONTRACT - win.max()],
        ['next year +15%', grown, CONTRACT - grown.max()],
        ['+ lunch waves', m1, CONTRACT - m1.max()],
        ['+ cooling to morning', m2, CONTRACT - m2.max()]]
out = []
for name, load, margin in rows:
    out.append([name, round(load.sum() / 1000, 1), round(load.max()),
                round(margin),
                round(load.mean() / load.max(), 2)])
df = pd.DataFrame(out, columns=['scenario', 'energy MWh', 'peak kW',
                                'margin to contract', 'daily LF'])
print(df.to_string(index=False))
print('---')
print("energy identical in every scenario:",
      round(grown.sum()) == round(m1.sum()) == round(m2.sum()),
      "(load moved, not removed)")
 
