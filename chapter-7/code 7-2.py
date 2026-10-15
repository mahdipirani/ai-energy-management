import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
rng = np.random.default_rng(701)          # same seed -> same month
 
# --- same synthetic site load as code 7-1 (Mehr 1405) --------------------
idx = pd.date_range('2026-09-23', periods=30 * 96, freq='15min')
 
def site_kw(ts):
    h = ts.hour + ts.minute / 60
    wd = ts.weekday()
    if wd == 4:
        return 400 + 25 * np.sin((h - 6) * np.pi / 12) ** 2
    if wd == 3:
        if h < 7:   return 520
        if h < 13:  return 1000
        if h < 22:  return 610
        return 540
    if h < 6:    return 520
    if h < 7:    return 520 + 600 * (h - 6)
    if h < 16:   return 1120
    if h < 18:   return 900
    if h < 22:   return 700
    return 560
 
kw = pd.Series([site_kw(ts) for ts in idx], index=idx)
kw = (kw * (1 + rng.normal(0, 0.03, len(kw)))).round(0)
 
# --- hour x weekday heat map ------------------------------------------------
pivot = (kw.groupby([kw.index.hour, kw.index.weekday]).mean()
          .unstack())                     # rows: hour, cols: weekday 0..6
pivot = pivot[[5, 6, 0, 1, 2, 3, 4]]      # reorder: Sat, Sun, Mon..Fri
pivot.columns = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']
 
fig, ax = plt.subplots(figsize=(10, 4.6))
im = ax.imshow(pivot.values, aspect='auto', origin='lower',
               cmap='YlGnBu')
ax.set_xticks(range(7))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(0, 24, 2))
ax.set_xlabel('weekday')
ax.set_ylabel('hour of day')
ax.set_title('Average load (kW): hour x weekday, Mehr 1405')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('kW')
r, c = np.unravel_index(np.argmax(pivot.values), pivot.shape)
ax.scatter(c, r, marker='o', s=120, facecolors='none',
           edgecolors='#B46C16', linewidths=2)
plt.tight_layout()
plt.show()
 
print('--- selected hours (kW) ---')
print(pivot.loc[[3, 7, 12, 17, 21]].round(0).to_string())
print(f"peak cell   : {pivot.values[r, c]:.0f} kW "
      f"({pivot.columns[c]} {r:02d}:00-{r+1:02d}:00)")
print(f"Fri 00-06h  : {pivot.loc[0:6, 'Fri'].mean():.0f} kW "
      f"(site at rest)")
print(f"Mon 07-16h  : {pivot.loc[7:16, 'Mon'].mean():.0f} kW")
 
