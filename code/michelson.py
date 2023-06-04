import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
import uncertainties
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (9,6)

wavelength = 0.6328 # micrometer

# Exercise 1: Error of the translation unit

m_translation_error = np.array([6.7, 7.4, 6.9, 6.9, 7.1, 6.9, 7.1, 7.2, 7.1, 6.9])
std_err = np.std(m_translation_error) / np.sqrt(len(m_translation_error))

# Exercise 2: Calibration of the translation unit

m_translation_calibration = np.array([7.1, 13.8, 19.9, 26.1, 33.6, 40.5, 47.5, 54.1, 61.1, 69.0])
m_translation_calibration_differences = []
m_translation_calibration_differences.append(ufloat(7.1, std_err))
for i in range(1, len(m_translation_calibration)):
    m_translation_calibration_differences.append(ufloat(m_translation_calibration[i] - m_translation_calibration[i - 1], np.sqrt(std_err**2 + 0.1**2)))
m_translation_calibration_differences = np.array(m_translation_calibration_differences)

m_translation_calibration = [sum(m_translation_calibration_differences[0:i]) for i in range(1, len(m_translation_calibration_differences) + 1)]

print(m_translation_calibration_differences)

fig, ax1 = plt.subplots()
x = unp.nominal_values(m_translation_calibration)
y = np.arange(20, 220, 20) * wavelength / 2
xerror = unp.std_devs(m_translation_calibration)

ax1.errorbar(x, y, xerr=xerror, fmt=".", label="Measurement data")
ax1.set_xlabel(r"Number of micrometer screw ticks $R$ in u")
ax1.set_ylabel(r"Theoretical discplacement $d_m = m \cdot \lambda / 2$")
ax1.set_title("Calibration of the translation unit")
ax1.grid(linewidth = 0.2)

def linear(x, a, c):
     return a*x + c

popt, pcov = curve_fit(linear, x, y, sigma=xerror, absolute_sigma=True)

gradient = ufloat(popt[0], np.sqrt(np.diag(pcov))[0])
print(gradient)

x_fit = np.linspace(min(x), max(x), 1000)
ax1.plot(x_fit, linear(x_fit, popt[0], popt[1]), label="Fit function $y = a \cdot R + c$")
ax1.legend(loc="upper left")

# plotting the conversion factors

y = (20 * wavelength / 2) / m_translation_calibration_differences
print("as")
for m in y:
    print('{:.2u}'.format(m))

fig, ax2 = plt.subplots()
ax2.errorbar(unp.nominal_values(m_translation_calibration_differences), unp.nominal_values(y), yerr=unp.std_devs(y), fmt=".", label="Ratio $d_m/R$")
ax2.set_xlabel(r"Number of micrometer screw ticks $R$ in u")
ax2.set_ylabel(r"Conversion factor $d_m/R$")
ax2.set_title("Calibration of the translation unit")
ax2.grid(linewidth = 0.2)
ax2.axhline(y=uncertainties.nominal_value(gradient), color='r', linestyle='-', label="Gradient of the fit function")
ax2.legend()

plt.show()