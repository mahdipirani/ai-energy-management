import pandas as pd
 
# --- one typical working day, hourly average load (kW) ----------------
hours = list(range(24))
kw = [430, 425, 420, 418, 422, 430, 470, 850, 920, 930, 925, 905,
      790, 915, 925, 920, 860, 760, 750, 740, 715, 690, 600, 500]
 
# --- time-of-use periods (industrial tariff) --------------------------
def period(h):
    if h < 7:                       # off-peak  00:00-07:00
        return 'off-peak'
    if 17 <= h < 23:                # peak      17:00-23:00
        return 'peak'
    return 'mid'                    # mid-load  07:00-17:00, 23:00-24:00
 
tariff = {'off-peak': 18.0, 'mid': 32.0, 'peak': 65.0}  # M-IRR / MWh
 
df = pd.DataFrame({'hour': hours, 'kw': kw})
df['period'] = df['hour'].map(period)
df['kwh'] = df['kw']                  # 1-hour slots -> kWh
 
g = df.groupby('period').agg(
    hours=('hour', 'count'), energy=('kwh', 'sum'))
g['energy_share_pct'] = (100 * g['energy'] / g['energy'].sum()).round(1)
g['tariff'] = g.index.map(tariff)
g['cost_mirr'] = (g['energy'] / 1000 * g['tariff']).round(1)
g['cost_share_pct'] = (100 * g['cost_mirr'] /
                       g['cost_mirr'].sum()).round(1)
print(g[['hours', 'energy', 'energy_share_pct', 'tariff',
         'cost_mirr', 'cost_share_pct']].to_string())
 
total_cost = g['cost_mirr'].sum()
print('---')
print(f"Total day cost        : {total_cost:.0f} M-IRR "
      f"(~{26*total_cost/1000:.1f} G-IRR per month)")
 
# --- scenario: shift 20% of peak energy to off-peak ---------------------
moved = 0.20 * g.loc['peak', 'energy']              # kWh
saving_day = moved / 1000 * (tariff['peak'] - tariff['off-peak'])
print(f"Shift 20% peak->off   : {saving_day:.0f} M-IRR/day "
      f"(~{26*saving_day/1000:.2f} G-IRR per month)")
 
