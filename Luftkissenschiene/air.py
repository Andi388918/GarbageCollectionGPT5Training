import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import uncertainties
import matplotlib.pyplot as plt

# measurements #######################################################################

placement_error = 0.5    # cm
time_error = 1e-4   # s
x0 = ufloat(186.52, placement_error)    # cm
x1 = ufloat(120, placement_error)       # cm
x0_instantaneous = ufloat(192, 0.3) # cm
s = x0 - x1
distances = unp.uarray([100, 84, 68, 52, 36], [np.sqrt(2) * placement_error])    # cm

times_for_distances = [   # cm
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

# functions ##########################################################################

def instantaneous_velocity(x, acceleration):
    distance, s = x
    return (2*acceleration*s)**(1/2) * 1/2 * ((1 + distance/(2*s))**(1/2) + (1 - distance/(2*s))**(1/2))

######################################################################################

# mean velocities, instantaneous velocity calculation ################################

# calculate the mean velocities from the distances and times recorded
mean_velocities = [distance / times_mean_velocity for distance, times_mean_velocity in zip(distances, times_for_distances)]   # cm/s

xerr = unp.std_devs(distances)  # cm / extract the uncertainties from the ditances
y = [np.mean(unp.nominal_values(mean_velocity_batch)) for mean_velocity_batch in mean_velocities]   # cm/s / calculate the mean velocity of each batch
yerr = [np.std(unp.nominal_values(mean_velocity_batch)) / len(mean_velocity_batch) for mean_velocity_batch in mean_velocities]  # cm/s / error is standard deviation of batch

popt, pcov = curve_fit(instantaneous_velocity, (unp.nominal_values(distances), [uncertainties.nominal_value(s)] * len(distances)), y, sigma=yerr, absolute_sigma=True)  # fit to measurement values
a = ufloat(popt[0], pcov[0])    # cm/s^2 / extract acceleration fit parameter

print(a)

x_values_fit = np.linspace(0, 100, 1000)    # cm
plt.plot(x_values_fit, unp.nominal_values(instantaneous_velocity((x_values_fit, np.array([s] * len(x_values_fit))), a)))    # plot the fit function

h = ufloat(1.5, 0.1)    # cm

# 0.015 m erhöhung bei l = 1 m länge
l = 100 # cm
g = a / unp.sin(unp.arctan(h / l)) / 100    # m/s^2

print(f"g (extrapolated from mean velocities) = {g}")

plt.errorbar(unp.nominal_values(distances), y, xerr, yerr, fmt = '.')

instantaneous_velocity_fit = instantaneous_velocity((0, s), a)  # the y-intercept of the fit function is equal to the instantaneous velocity
instantaneous_velocities_measured = unp.uarray([47.6, 47.6, 47.3, 47.3, 47.3], [0.1] * 5)    # these are the measurment values for the 'instantaneous velocity'

# since we measured the mean velocities and instant. vel. at different positions and the velocity increases 
# approx. linearly with distance, we can compare them by multipling the higher velocity with a correciton factor
# that depends on the distances
corrected_instantaneous_velocities = unp.nominal_values(instantaneous_velocities_measured * x0/x0_instantaneous)
mean_instantaneous_velocity_measured = np.mean(corrected_instantaneous_velocities)
std_dev_instantaneous_velocity_measured = np.std(corrected_instantaneous_velocities) / len(corrected_instantaneous_velocities)
instantaneous_velocity_measured = ufloat(mean_instantaneous_velocity_measured, std_dev_instantaneous_velocity_measured)

print(f"instantaneous velocity from fit = {instantaneous_velocity_fit}")
print(f"instantaneous velocity from measurement = {instantaneous_velocity_measured}")

######################################################################################

# alternative calculation of g: measuring two different velocities and using the potential energy difference
# of the gravitational field to calculate the acceleration ###########################

velocities_sensor_1 = unp.uarray([7.7] * 10, [0.1] * 10)    # cm/s
velocities_sensor_2 = unp.uarray([10.9] * 10, [0.1] * 10)   # cm/s

x1 = ufloat(140, placement_error)   # cm
x2 = ufloat(90, placement_error)    # cm

mean_velocity_sensor_1 = ufloat(np.mean(unp.nominal_values(velocities_sensor_1)), 0)
mean_velocity_sensor_2 = ufloat(np.mean(unp.nominal_values(velocities_sensor_2)), 0)

a2 = (mean_velocity_sensor_1 ** 2 - mean_velocity_sensor_2 ** 2) / (2 * (x1 - x2))
print(a2)

######################################################################################

# elastic collision ##################################################################

######################################################################################

# show plot ##########################################################################

plt.show()