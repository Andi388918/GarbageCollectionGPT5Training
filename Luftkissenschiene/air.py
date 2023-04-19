import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
import uncertainties
import matplotlib.pyplot as plt

# functions ##########################################################################

def get_speed_with_relative_correction(measurement, correction_interval = [0.01, 0.02]):
    correction1 = measurement * (1 - correction_interval[0])
    correction2 = measurement * (1 - correction_interval[1])
    uncertainty = 1/2 * (np.abs(uncertainties.nominal_value(correction1) - uncertainties.nominal_value(correction2)) + uncertainties.std_dev(correction1) + uncertainties.std_dev(correction2))
    return ufloat(np.mean([uncertainties.nominal_value(correction1), uncertainties.nominal_value(correction2)]), uncertainty)

def get_speed_with_random_error(measurement):
    return ufloat(measurement, (0.1**2 + (3 * 10**(-5) * measurement ** 2) ** 2) ** (1/2))   # cm/s

def get_speed_with_total_error(measurement):
    return get_speed_with_relative_correction(get_speed_with_random_error(measurement))

def get_speeds_with_total_uncertainties(measurements):
    return np.array([get_speed_with_total_error(measurement) for measurement in measurements])

def instantaneous_velocity(x, acceleration):
    distance, s = x
    return (2*acceleration*s)**(1/2) * 1/2 * ((1 + distance/(2*s))**(1/2) + (1 - distance/(2*s))**(1/2))

######################################################################################

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

######################################################################################

# mean velocities, instantaneous velocity calculation ################################

# calculate the mean velocities from the distances and times recorded
mean_velocities = [distance / times_mean_velocity for distance, times_mean_velocity in zip(distances, times_for_distances)]   # cm/s

x_err = unp.std_devs(distances)  # cm / extract the uncertainties from the distances
y_with_uncertainties = [np.mean(mean_velocity_batch) for mean_velocity_batch in mean_velocities]   # cm/s / calculate the mean velocity of each batch
y = unp.nominal_values(y_with_uncertainties)
y_err = unp.std_devs(y_with_uncertainties)  # cm/s

popt, pcov = curve_fit(instantaneous_velocity, (unp.nominal_values(distances), [uncertainties.nominal_value(s)] * len(distances)), y, sigma=y_err, absolute_sigma=True)  # fit to measurement values
a1 = ufloat(popt[0], pcov[0])    # cm/s^2 / extract acceleration fit parameter

x_values_fit = np.linspace(0, 100, 1000)   # cm
plt.plot(np.square(x_values_fit), unp.nominal_values(instantaneous_velocity((x_values_fit, np.array([s] * len(x_values_fit))), a1)), label="Fit function")    # plot the fit function
plt.xlabel(r"Distance $D^2$ in cm$^2$")
plt.ylabel(r"Mean velocity $\overline{\langle v \rangle}$ in cm/s")
plt.title("Measurement of the mean velocity")
plt.grid(color='grey', linestyle='-', linewidth=0.4)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.text(2000, 41.5, r"$a_1 = $" + '{:.2uS}'.format(a1) + r" cm/s$^2$", fontsize=10)

h = ufloat(1.5, 0.1)    # cm

# 0.015 m elevation at l = 1 m length
l = 100 # cm
g1 = a1 / unp.sin(unp.arctan(h / l)) / 100    # m/s^2

print(f"g (extrapolated from mean velocities) = {g1} m/s^2")

plt.errorbar(unp.nominal_values(distances) ** 2, y, xerr = x_err, yerr = y_err, fmt = '.', label="Measurement data")
plt.legend()

print(a1)
instantaneous_velocity_fit = instantaneous_velocity((0, s), a1)  # the y-intercept of the fit function is equal to the instantaneous velocity
instantaneous_velocities_measured = get_speeds_with_total_uncertainties([47.6, 47.6, 47.3, 47.3, 47.3])    # these are the measurment values for the 'instantaneous velocity'

# since we measured the mean velocities and instant. vel. at different positions and the velocity increases 
# approx. linearly with distance, we can compare them by multipling the higher velocity with a correction factor
# that depends on the distances
instantaneous_velocity_measured = np.mean(instantaneous_velocities_measured)

print(f"instantaneous velocity from fit = {instantaneous_velocity_fit} cm/s")
print(f"instantaneous velocity from measurement = {instantaneous_velocity_measured} cm/s")

a3 = (instantaneous_velocity_measured**2)/(2*(x0_instantaneous - x1))
print(f"a3 (from instant. vel.) = {a3}")


print(instantaneous_velocity_measured)
print(a3)

######################################################################################

# alternative calculation of g: measuring two different velocities and using the potential energy difference
# of the gravitational field to calculate the acceleration ###########################

velocities_sensor_1 = get_speeds_with_total_uncertainties([7.7] * 10)    # cm/s
velocities_sensor_2 = get_speeds_with_total_uncertainties([10.9] * 10)   # cm/s

x1 = ufloat(140, placement_error)   # cm
x2 = ufloat(90, placement_error)    # cm

mean_velocity_sensor_1 = np.mean(velocities_sensor_1)
mean_velocity_sensor_2 = np.mean(velocities_sensor_2)

a2 = (mean_velocity_sensor_1 ** 2 - mean_velocity_sensor_2 ** 2) / (2 * (x2 - x1))  # cm/s^2

print("a2", a2)

g2 = a2 / unp.sin(unp.arctan(h / l)) / 100  # m/s^2
print(f"g (calculated from inst. vel. difference) = {g2} m/s^2")

g3 = a3 / unp.sin(unp.arctan(h / l)) / 100  # m/s^2
print(f"g3 (calculated from inst. vel. difference) = {g3} m/s^2")

######################################################################################

# collisions #########################################################################

# unweighted elastic collision

distance = ufloat(40, 1)    # cm
m1 = ufloat(207, 1) # g
m2 = ufloat(205, 1) # g

v1_i = get_speeds_with_total_uncertainties([43.4, 45.8, 27.3, 34.8, 38.4]) # cm/s
v2_i = get_speeds_with_total_uncertainties([-41.8, -40.3, -27.1, -42.5, -35.9])    # cm/s
v1_a = get_speeds_with_total_uncertainties([-38.7, -37.7, -26.3, -37.1, -34.2])    # cm/s
v2_a = get_speeds_with_total_uncertainties([15.7, 44.0, 26.6, 33.8, 37.5]) # cm/s

vsp_i = (m1 * v1_i + m2 * v2_i) / (m1 + m2)    # cm/s
vsp_a = (m1 * v1_a + m2 * v2_a) / (m1 + m2)    # cm/s
vrel_i = v1_i - v2_i
vrel_a = v1_a - v2_a

delta_vsp = vsp_a - vsp_i   # cm/s
print(f"elastic collision (unweighted) vsp_a - vsp_i:")
print(delta_vsp)

eta = (vrel_a/vrel_i)**2
print(f"elastic collision (unweighted) elasticity parameter:")
print(eta)

# weighted elastic collision

m1 = ufloat(406, 1) # g
m2 = ufloat(205, 1) # g

v1_i = get_speeds_with_total_uncertainties([38.9, 27.4, 33.8, 48, 50]) # cm/s
v2_i = get_speeds_with_total_uncertainties([-54.3, -32.7, -42.9, -52, -41.8])    # cm/s
v1_a = get_speeds_with_total_uncertainties([-21, -10, -15.9, -14.6, -6.3])    # cm/s
v2_a = get_speeds_with_total_uncertainties([64.9, 44.6, 56.4, 75.1, 67.1]) # cm/s

vsp_i = (m1 * v1_i + m2 * v2_i) / (m1 + m2)    # cm/s
vsp_a = (m1 * v1_a + m2 * v2_a) / (m1 + m2)    # cm/s
vrel_i = v1_i - v2_i
vrel_a = v1_a - v2_a

delta_vsp = vsp_a - vsp_i   # cm/s
print(f"elastic collision (weighted) vsp_a - vsp_i:")
print(delta_vsp)

eta = (vrel_a/vrel_i)**2
print(f"elastic collision (weighted) elasticity parameter:")
print(eta)

# inelastic collision

distance = ufloat(40, 1)    # cm
m1 = ufloat(207, 1) # g
m2 = ufloat(205, 1) # g

v1_i = get_speeds_with_total_uncertainties([55.8, 64.5, 61.7, 58.8, 52.6]) # cm/s
v2_i = get_speeds_with_total_uncertainties([-66.2, -61.7, -49, -45.6, -74])    # cm/s
v_a = get_speeds_with_total_uncertainties([-4, 1.9, 5.8, 6.2, -9.5])  # cm/s

vsp_i = (m1 * v1_i + m2 * v2_i) / (m1 + m2)    # cm/s
vsp_a = v_a    # cm/s
vrel_i = v1_i - v2_i
vrel_a = v_a

delta_vsp = vsp_a - vsp_i   # cm/s
print(f"inelastic collision v_a - vsp_i:")
print(delta_vsp)

eta = (vrel_a/vrel_i)**2
print(f"inelastic collision elasticity parameter:")
print(eta)

######################################################################################

# show plot ##########################################################################

plt.show()