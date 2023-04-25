import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import uncertainties
import matplotlib.pyplot as plt

def print_uarray_pretty(uarray):
    print("[")
    for value in uarray:
        print('{:.2uS}'.format(value))
    print("]")

def get_voltage_with_error(voltage1, voltage2):
    voltage1 = ufloat(voltage1, np.abs(voltage1 * 0.008) + 0.001)
    voltage2 = ufloat(voltage2, np.abs(voltage2 * 0.008) + 0.001)
    return np.mean([voltage1, voltage2])

start_temperature = 140 # celsius
end_temperature = 360 # celsius
step_size = 5 # celsius

angle = 10 # degrees
T_D = 298 # K

temperatures = np.arange(start_temperature, end_temperature + step_size, step_size)
temperatures_kelvin = temperatures + 273.15

voltages = [
    (2.116, 2.116),
    (2.005, 1.996),
    (1.924, 1.906),
    (1.842, 1.829),
    (1.772, 1.757),
    (1.700, 1.685),
    (1.634, 1.622),
    (1.572, 1.561),
    (1.509, 1.496),
    (1.451, 1.439),
    (1.388, 1.375),
    (1.328, 1.318),
    (1.267, 1.261),
    (1.213, 1.202),
    (1.161, 1.151),
    (1.104, 1.092),
    (1.049, 1.040),
    (0.997, 0.985),
    (0.946, 0.936),
    (0.898, 0.890),
    (0.851, 0.842),
    (0.803, 0.795),
    (0.760, 0.750),
    (0.714, 0.706),
    (0.671, 0.662),
    (0.628, 0.621),
    (0.589, 0.582),
    (0.547, 0.540),
    (0.506, 0.501),
    (0.462, 0.460),
    (0.432, 0.424),
    (0.395, 0.389),
    (0.359, 0.356),
    (0.327, 0.321),
    (0.294, 0.288),
    (0.243, 0.247),
    (0.225, 0.219),
    (0.193, 0.187),
    (0.163, 0.156),
    (0.132, 0.127),
    (0.107, 0.102),
    (0.076, 0.071),
    (0.055, 0.049),
    (0.027, 0.022),
    (0.001, 0.008)
]
""" ,
    (0, 0),
    (-0.048, 0.048),
    (-0.005, -0.069),
    (-0.088, -0.093),
    (-0.111, -0.111),
    (-0.127, -0.131),
    (-0.149, -0.152),
    (-0.167, -0.168)
]
 """

voltages.reverse()
voltages = np.array([get_voltage_with_error(voltage_range[0], voltage_range[1]) for voltage_range in voltages])

#############################################################

plt.rcParams["figure.figsize"] = (9,6)

x = temperatures_kelvin
y = unp.nominal_values(voltages)
y_err = unp.std_devs(voltages)

fig1, ax1 = plt.subplots()
ax1.errorbar(x, y, xerr=0, yerr=y_err, fmt="none", label="Measurement data", capsize=3)
ax1.set_title("Voltage vs. Temperature Measurements\n of Black Body Radiation")
ax1.set_xlabel("Temperature T in K")
ax1.set_ylabel("Voltage U in mV")
ax1.legend()

ax1.set_yticks(np.arange(0, 2, 0.25), minor=True)
ax1.set_xticks(np.arange(400, 660, 25), minor=True)
ax1.grid(which='both', linewidth = 0.2)

#############################################################

voltages_logarithmic = unp.log(voltages)

x = np.log(temperatures_kelvin)
y = unp.nominal_values(voltages_logarithmic)
y_err = unp.std_devs(voltages_logarithmic)

fig2, ax2 = plt.subplots()
ax2.errorbar(x, y, xerr=0, yerr=y_err, fmt="none", label="Measurement data", capsize=3)
ax2.set_title("Logarithmic Relationship Between Voltage \nand Temperature in Black Body Radiation")
ax2.set_xlabel("log(T) [dimensionless]")
ax2.set_ylabel("log(U) [dimensionless]")
ax2.legend()
ax2.set_yticks(np.arange(-6, 1, 0.5), minor=True)
ax2.set_xticks(np.arange(6, 6.5, 0.025), minor=True)
ax2.grid(which='both', linewidth = 0.2)

##############################################################

differential_quotients = []
for i in range(1, len(voltages)):
    differential_quotient = (voltages[i] - voltages[i - 1]) / 5
    differential_quotients.append(differential_quotient)
differential_quotients = np.array(differential_quotients)
differential_quotients_logarithmic = unp.log(differential_quotients)

x = x[:-1]
y = unp.nominal_values(differential_quotients_logarithmic)
y_err = unp.std_devs(differential_quotients_logarithmic)

fig3, ax3 = plt.subplots()
ax3.errorbar(x, y, xerr=0, yerr=y_err, fmt="none", label="Measurement data", capsize=3)
ax3.set_title("Voltage Gradient vs. Logarithmic\n Temperature in Black Body Radiation")
ax3.set_xlabel("log(T) [dimensionless]")
ax3.set_ylabel(r"$\frac{dU}{dT}$ in mV/K")
ax3.grid()
ax3.set_yticks(np.arange(-5.75, -3.5, 0.25), minor=True)
ax3.set_xticks(np.arange(6, 6.5, 0.025), minor=True)
ax3.grid(which='both', linewidth = 0.2)

def diagram_c_fit(LOG_T, c):
    return 3 * LOG_T + c

popt, pcov = curve_fit(diagram_c_fit, x, y, sigma=y_err, absolute_sigma=True)
s = popt[0]
print(pcov)
x_data = np.linspace(min(x), max(x), 100)
ax3.plot(x_data, diagram_c_fit(x_data, s), color="red", label=r"Fit function $f(x) = 3 \cdot log(T) + c$")
ax3.legend(loc = "lower right")

s_with_uncertainty = ufloat(s, np.sqrt(pcov[0]))

textstr = r'$c=$' + f"{'{:.2uS}'.format(s_with_uncertainty)} mV/K"
ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top')

plt.show()