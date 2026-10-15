import numpy as np
import matplotlib.pyplot as plt

# --- Monthly electricity of Pars-Ayandeh (MWh), 24 months -----------
kwh_1403 = [512, 516, 523, 523, 531, 536, 527, 519,
            516, 523, 526, 528]
kwh_1404 = [511, 516, 522, 524, 533, 539, 531, 523,
            521, 529, 533, 535]

total_1, total_2 = sum(kwh_1403), sum(kwh_1404)
print(f"Total 1403 : {total_1:,} MWh")
print(f"Total 1404 : {total_2:,} MWh")
print(f"YoY change : {100*(total_2-total_1)/total_1:.1f} %")

# --- Average industrial tariff (million IRR per MWh) ---------------
tariff_1, tariff_2 = 28.0, 39.5          # +41 %
bill_1 = total_1 * tariff_1
bill_2 = total_2 * tariff_2
print(f"Bill 1403  : {bill_1:,.0f} M-IRR")
print(f"Bill 1404  : {bill_2:,.0f} M-IRR")
print(f"Bill change: {100*(bill_2-bill_1)/bill_1:.1f} %")

# --- 24-month trend chart --------------------------------------------
months = np.arange(1, 25)
kwh_all = np.array(kwh_1403 + kwh_1404)
plt.figure(figsize=(10, 4.2))
plt.plot(months, kwh_all, marker='o', color='#0B4E6B')
plt.axvline(12.5, color='#B46C16', linestyle='--')
plt.text(6.3, 537, 'year 1403', color='#0B4E6B')
plt.text(18.5, 537, 'year 1404', color='#0B4E6B')
plt.xlabel('month number (1 = Farvardin 1403)')
plt.ylabel('electricity (MWh)')
plt.title('Pars-Ayandeh monthly electricity consumption')
plt.tight_layout()
plt.show()