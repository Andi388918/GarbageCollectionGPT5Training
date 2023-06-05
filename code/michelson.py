import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
import uncertainties
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (9,6)

wavelength = ufloat(0.6328, 0.005) # micrometer

# Exercise 1: Error of the translation unit

m_translation_error = np.array([6.7, 7.4, 6.9, 6.9, 7.1, 6.9, 7.1, 7.2, 7.1, 6.9])
std_err = np.std(m_translation_error) / np.sqrt(len(m_translation_error))
total_err = np.sqrt(std_err**2 + 0.1**2)

# Exercise 2: Calibration of the translation unit

m_translation_calibration = unp.uarray([7.1, 13.8, 19.9, 26.1, 33.6, 40.5, 47.5, 54.1, 61.1, 69.0], [np.sqrt(i * total_err ** 2) for i in range(1, 11)])

displacements = unp.uarray(np.arange(20, 220, 20), np.linspace(0.5, 5, 10))
displacements *= wavelength / 2

fig, ax1 = plt.subplots()
x = unp.nominal_values(m_translation_calibration)
y = unp.nominal_values(displacements)
yerror = unp.std_devs(displacements)
xerror = unp.std_devs(m_translation_calibration)

ax1.errorbar(x, unp.nominal_values(y), xerr=xerror, yerr=yerror, fmt=".", label="Measurement data")
ax1.set_xlabel(r"Number of micrometer screw ticks $R$ in u")
ax1.set_ylabel(r"Mirror discplacement $d_m = m \cdot \lambda / 2$ in μm")
ax1.set_title("Calibration of the translation unit")
ax1.grid(linewidth = 0.2)

def linear(x, a, c):
     return a*x + c

popt, pcov = curve_fit(linear, x, y, sigma=yerror, absolute_sigma=True)

gradient = ufloat(popt[0], np.sqrt(np.diag(pcov))[0])

x_fit = np.linspace(min(x), max(x), 1000)
ax1.plot(x_fit, linear(x_fit, popt[0], popt[1]), label="Fit function $y = a \cdot R + c$")
ax1.legend(loc="upper left")

# plotting the conversion factors

translation_differences = unp.uarray(np.diff(unp.nominal_values(m_translation_calibration)), total_err)
translation_differences = list(translation_differences)
translation_differences.insert(0, m_translation_calibration[0])
translation_differences = np.array(translation_differences)

x = unp.nominal_values(translation_differences)
xerror = unp.std_devs(translation_differences)

conversion_factors = (20 * wavelength / 2) / translation_differences

""" for m in conversion_factors:
    print('{:.2u}'.format(m)) """

y = unp.nominal_values(conversion_factors)
yerror = unp.std_devs(conversion_factors)

fig, ax2 = plt.subplots()
ax2.errorbar(x, y, xerr=xerror, yerr=yerror, fmt=".", label="Ratio $d_{20}/\Delta R$")
ax2.set_xlabel(r"Number of micrometer screw ticks $\Delta R$ in u")
ax2.set_ylabel(r"Conversion factor $d_{20}/\Delta R$")
ax2.set_title("Calibration of the translation unit")
ax2.grid(linewidth = 0.2)
ax2.axhline(y=uncertainties.nominal_value(np.mean(conversion_factors)), color='r', linestyle='-', label="Mean conversion factor")
ax2.legend()

# refraction index of air

# print(np.mean(conversion_factors))

length_of_chamber = ufloat(4.44, 0.10) * 10**(-2) # m
pressure_air_outside = ufloat(1016, 8) # mbar

pressures = pressure_air_outside - unp.uarray([600, 565, 505, 480, 450, 420, 385, 355, 320, 300, 270, 240, 210, 185, 150, 120, 105, 65, 50, 40], 8)
interference_ring_counts = unp.uarray(list(range(20)), 0.5)
delta_n = interference_ring_counts * wavelength / (2 * length_of_chamber * 10**(6))

""" print("-")
for m in delta_n:
    print('{:.1u}'.format(m))
print("-") """

x = unp.nominal_values(pressures)
y = unp.nominal_values(delta_n)
xerror = unp.std_devs(pressures)
yerror = unp.std_devs(delta_n)

fig, ax3 = plt.subplots()
ax3.errorbar(x, y, xerr=xerror, yerr=yerror, fmt=".", label="Messdaten")
ax3.set_xlabel(r"Air pressure of chamber $\rho$ in mbar")
ax3.set_ylabel(r"Refractive index $n(\rho) - 1$")
ax3.set_title("Refractive index of air")
ax3.grid(linewidth = 0.2)

popt, pcov = curve_fit(linear, x, y, sigma=yerror, absolute_sigma=True)
a = ufloat(popt[0], np.sqrt(np.diag(pcov))[0])
x_fit = np.linspace(min(x), max(x), 1000)
ax3.plot(x_fit, linear(x_fit, popt[0], popt[1]), label=r"Fit function $y = a \cdot \rho + c$")
ax3.legend(loc="upper left")

print(a)

plt.show()