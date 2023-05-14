import glob
import re
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
import uncertainties
from scipy.optimize import curve_fit

import os
script_directory = os.path.dirname(os.path.realpath(__file__))

# Exercise 1: Activity vs voltage of the geiger counter

file_pattern = os.path.join(script_directory, "data_rs\A1\*V_15sec.txt")
files = glob.glob(file_pattern)

measurements = {}

for directory_name in files:
    filename = directory_name.split("\\")[-1]

    match = re.match(r"\d+", filename)
    number = int(match.group(0))
    with open(directory_name, "r") as file:
        lines = file.readlines()
        activities = []
        t, a = map(int, lines[4].split())
        activities.append(ufloat(a, 4 * np.sqrt(a/4)))
        previous_t = t
        for line in lines[5:]:
            if line.strip() != "":
                t, a = map(int, line.split())
                if t - previous_t > 1e4:
                    activities.append(ufloat(a, 4 * np.sqrt(a/4)))
                previous_t = t
        measurements[number] = np.array(activities)
    
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (9,6)
activities = np.array([np.mean(a) for a in list(measurements.values())])

fig, ax1 = plt.subplots()
ax1.errorbar(x=list(measurements.keys()), y=unp.nominal_values(activities), xerr=5, yerr=unp.std_devs(activities), fmt=".", label="Messdaten")
ax1.set_xlabel(r"Zählrohrspannung $U$ in V")
ax1.set_ylabel(r"Zählrate $A$ in min$^{-1}$")
ax1.set_title("Charakteristik der Geigerzählröhre")
ax1.grid(linewidth = 0.2)
ax1.legend(loc="upper left")

# Exercise 2: Background radiation

directory_name = os.path.join(script_directory, "data_rs\A2\Hintergrundmessung_1100V_15_sec.txt")
bgr_times = []
bgr_activities = []

with open(directory_name, "r") as file:
        lines = file.readlines()
        for line in lines[4:]:
             if line.strip() != "":
                t, a = map(int, line.split())
                bgr_times.append(t)
                bgr_activities.append(ufloat(a, 4 * np.sqrt(a/4)))

bgr_activities = np.array(bgr_activities)
mean_bgr_activity = np.mean(bgr_activities)
bgr_activities_15_sec_interval = bgr_activities/4

fig, ax2 = plt.subplots()
ax2.hist(unp.nominal_values(bgr_activities_15_sec_interval), density=True, bins=30)
ax2.set_xlabel(r"Zählrate $A$ in (15 s)$^{-1}$")
ax2.set_ylabel(r"Häufigkeit")
ax2.set_title(r"Statistik radioaktiver Zerfälle (Hintergrundstrahlung)")
ax2.grid(linewidth = 0.2)

# Exercise 3: Activity of Ba 137 *

directory_name = os.path.join(script_directory, "data_rs\A2\Ba_15sec_1100V.txt")
times = []
activities = []

with open(directory_name, "r") as file:
        lines = file.readlines()
        for line in lines[4:]:
             if line.strip() != "":
                t, a = map(int, line.split())
                times.append(t)
                activities.append(ufloat(a, 4 * np.sqrt(a/4)))

times = np.array(times)/1e3
activities = (np.array(activities) - mean_bgr_activity)/60

fig, ax3 = plt.subplots()
ax3.errorbar(x=times, y=unp.nominal_values(activities), xerr=0, yerr=unp.std_devs(activities), fmt=".", label="Messdaten")
ax3.set_xlabel(r"Zeit $t$ in s")
ax3.set_ylabel(r"Log. Aktivität $log(A)$ [dimensionslos]")
ax3.set_title(r"Zerfall des $^{137}$Ba$^*$ Isotops")
ax3.grid(linewidth = 0.2)
ax3.set_yscale('log')

def linear(x, a, c):
     return a*x + c

popt, pcov = curve_fit(linear, times, np.log(unp.nominal_values(activities)), sigma=unp.std_devs(activities), absolute_sigma=True)

lambda_c = -ufloat(popt[0], np.sqrt(np.diag(pcov))[0])
half_life = np.log(2)/lambda_c

x_fit = np.linspace(0, max(times), 1000)
ax3.plot(x_fit, np.exp(linear(x_fit, popt[0], popt[1])), label="Fitfunktion $y = -\lambda \cdot x + c$")
ax3.legend(loc="upper right")

print(f"half life = {half_life} s")

plt.show()