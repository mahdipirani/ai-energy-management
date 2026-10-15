import pandas as pd
 
# --- Monthly production (tons) and electricity (MWh) ----------------
tons_1403 = [848, 852, 860, 855, 870, 882, 875, 865,
             858, 862, 868, 875]
tons_1404 = [845, 848, 852, 850, 865, 878, 870, 860,
             852, 856, 862, 868]
kwh_1403 = [512, 516, 523, 523, 531, 536, 527, 519,
            516, 523, 526, 528]
kwh_1404 = [511, 516, 522, 524, 533, 539, 531, 523,
            521, 529, 533, 535]
 
df = pd.DataFrame({
    'year': [1403] * 12 + [1404] * 12,
    'tons': tons_1403 + tons_1404,
    'kwh':  kwh_1403 + kwh_1404,
})
df['enpi'] = (df['kwh'] / df['tons']).round(3)
 
print(df.head(6).to_string(index=False))
print('---')
for year, g in df.groupby('year'):
    print(f"Year {year}: mean EnPI = {g['enpi'].mean():.3f} MWh/ton")
