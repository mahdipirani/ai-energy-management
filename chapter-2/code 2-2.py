import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- same monthly data as Code 2-1 -----------------------------------
tons_1403 = [848, 852, 860, 855, 870, 882, 875, 865,
             858, 862, 868, 875]
tons_1404 = [845, 848, 852, 850, 865, 878, 870, 860,
             852, 856, 862, 868]
kwh_1403 = [512, 516, 523, 523, 531, 536, 527, 519,
            516, 523, 526, 528]
kwh_1404 = [511, 516, 522, 524, 533, 539, 531, 523,
            521, 529, 533, 535]

# --- 1) fit the baseline on year 1403 --------------------------------
base = pd.DataFrame({'tons': tons_1403, 'kwh': kwh_1403})
model = LinearRegression()
model.fit(base[['tons']], base['kwh'])
a, b = model.coef_[0], model.intercept_
print(f"Baseline model : kWh = {a:.3f} * tons + {b:.1f}")

# --- 2) compare year 1404 against the baseline ------------------------
mon = pd.DataFrame({'tons': tons_1404, 'kwh': kwh_1404})
mon['predicted'] = model.predict(mon[['tons']]).round(1)
mon['residual'] = mon['kwh'] - mon['predicted']
mon['cusum'] = mon['residual'].cumsum().round(1)
print(mon[['tons', 'kwh', 'predicted', 'cusum']]
      .to_string(index=False))

excess = mon['cusum'].iloc[-1]
print(f"Cumulative excess : {excess:.0f} MWh "
      f"({100*excess/mon['kwh'].sum():.1f} %)")

# --- 3) charts --------------------------------------------------------
plt.figure(figsize=(10, 4.2))
plt.scatter(base['tons'], base['kwh'], color='#0B4E6B',
            label='1403 (baseline)')
plt.scatter(mon['tons'], mon['kwh'], color='#B46C16',
            label='1404 (actual)')
xs = np.linspace(840, 885, 50).reshape(-1, 1)
plt.plot(xs, model.predict(xs), color='#0E8A6A', lw=2,
         label='baseline')
plt.xlabel('production (tons)')
plt.ylabel('electricity (MWh)')
plt.legend()
plt.title('Energy baseline: consumption vs production')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4.2))
x = np.arange(1, 13)
plt.bar(x, mon['cusum'], color='#B46C16', width=0.62)
plt.axhline(0, color='#0B4E6B', lw=1)
plt.xlabel('month of 1404')
plt.ylabel('cumulative excess (MWh)')
plt.title('CUSUM: drift away from the baseline')
plt.tight_layout()
plt.show()