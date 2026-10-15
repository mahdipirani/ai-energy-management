import pandas as pd
import matplotlib.pyplot as plt
 
# --- 8 months of sub-meter data (Farvardin .. Aban 1405) ------------
months = list(range(1, 9))                 # 1 = Farvardin 1405
main_meter = [528, 531, 536, 538, 543, 552, 544, 539]   # MWh
sub = pd.DataFrame({
    'furnace':         [208, 210, 214, 215, 220, 224, 221, 217],
    'production_line': [148, 150, 151, 153, 156, 158, 157, 154],
    'compressed_air':  [ 52,  53,  55,  56,  58,  60,  59,  57],
    'hvac_chiller':    [ 38,  40,  48,  55,  62,  66,  63,  55],
    'lighting':        [ 24,  24,  23,  22,  21,  21,  21,  22],
    'office_misc':     [ 18,  18,  19,  19,  20,  20,  19,  19],
}, index=months)
 
# --- metering coverage check ------------------------------------------
sub['sub_total'] = sub.sum(axis=1)
sub['main'] = main_meter
sub['unaccounted'] = sub['main'] - sub['sub_total']
sub['coverage_pct'] = (100 * sub['sub_total'] / sub['main']).round(1)
 
print(sub[['main', 'sub_total', 'unaccounted', 'coverage_pct']]
      .to_string())
print('---')
avg_cov = sub['coverage_pct'].mean()
print(f"Average metering coverage : {avg_cov:.1f} %")
print(f"Worst month unaccounted   : "
      f"{sub['unaccounted'].max():.0f} MWh "
      f"(month {sub['unaccounted'].idxmax()})")
 
# --- load map figure ----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.6))
bottom = None
colors = ['#0B4E6B', '#11557A', '#0E8A6A', '#B46C16',
          '#5FA8C0', '#8A97A6']
for (col, c) in zip(['furnace', 'production_line', 'compressed_air',
                     'hvac_chiller', 'lighting', 'office_misc'], colors):
    ax.bar(months, sub[col], bottom=bottom, color=c, width=0.62,
           label=col.replace('_', ' '))
    bottom = sub[col] if bottom is None else bottom + sub[col]
ax.plot(months, main_meter, color='#8B2E2E', lw=2.2, marker='D',
        ms=5, label='main meter')
ax.bar(months, sub['unaccounted'], bottom=sub['sub_total'],
       color='#8B2E2E', alpha=0.35, width=0.62,
       label='unaccounted')
ax.set_xlabel('month of 1405 (1 = Farvardin)')
ax.set_ylabel('energy (MWh)')
ax.set_title('Load map: sub-meters, main meter and unaccounted energy')
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.grid(alpha=0.25, ls=':', axis='y')
plt.tight_layout()
plt.show()
 
