import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

def instantaneous_velocity(D, a):
    S = 186.52 - 120
    return (2*a*S)**(1/2) * 1/2 * ((1 + D/(2*S))**(1/2) + (1 - D/(2*S))**(1/2))

sensor_placement_error = 5e-3
time_error = 1e-4

x0 = ufloat(186.52, 0.50)
x1 = ufloat(120, 0)
D = unp.uarray([100, 84, 68, 52, 36], [np.sqrt(2) * sensor_placement_error])

times_mean_velocities = [
        unp.uarray(
            [2.4437, 2.4400, 2.4438, 2.4407, 2.4405],
            [time_error] * 5
        ),
        unp.uarray(
            [1.9863, 1.9878, 1.9871, 1.9880, 1.9873],
            [time_error] * 5
        ),
        unp.uarray(
            [1.5856, 1.5827, 1.5839, 1.5828, 1.5843],
            [time_error] * 5
        ),
        unp.uarray(
            [1.1746, 1.1740, 1.1732, 1.1739, 1.1735],
            [time_error] * 5
        ),
        unp.uarray(
            [0.8098, 0.8095, 0.8091, 0.8086, 0.8098],
            [time_error] * 5
        )
]

mean_velocities = [distance / times_mean_velocity for distance, times_mean_velocity in zip(D, times_mean_velocities)]

import matplotlib.pyplot as plt

xerr = unp.std_devs(D)
y = [np.mean(unp.nominal_values(times)) for times in mean_velocities]
yerr = [np.std(unp.nominal_values(times)) / len(times) for times in mean_velocities]

popt, pcov = curve_fit(instantaneous_velocity, unp.nominal_values(D), y, sigma=yerr, absolute_sigma=True)
a = ufloat(popt[0], pcov[0])

x_values_fit = np.linspace(0, 100, 1000)
plt.plot(x_values_fit, unp.nominal_values(instantaneous_velocity(x_values_fit, a)))

# 15.2 cm/s^2
# 0.0152 m/s^2

# test

h = ufloat(0.015, 0.001)

# 0.015 m erhöhung bei 1 m länge
g = a/100 / unp.sin(unp.arctan(h))
print(g)

plt.errorbar(unp.nominal_values(D), y, xerr, yerr, fmt = '.')
plt.show()

instantaneous_velocity_fit = instantaneous_velocity(0, a)
print(instantaneous_velocity_fit)
instantaneous_velocities = unp.uarray([47.6, 47.6, 47.3, 47.3, 47.3], [0.1] * 5)

print(np.mean(instantaneous_velocity_fit) * 186.52/192)