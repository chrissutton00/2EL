import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['axes.formatter.useoffset'] = False

# Rabifit
def RabiFit(Data, Output = True):
    Time = Data[:,0]
    HeterodynSignal = Data[:,1]
    a_guess = (np.max(HeterodynSignal) - np.min(HeterodynSignal))/2
    phi0_guess = 0
    T_guess = Time[-1]/5
    c_guess = HeterodynSignal[-1]
    omega_guess = 6.28/(2*np.abs(Time[np.argmax(HeterodynSignal)] - Time[np.argmin(HeterodynSignal)]))

    popt, pcov = curve_fit(lambda t, a, T, c, omega, phi0: a * np.cos(omega*t+phi0) * np.exp(-t/T) + c, Time, HeterodynSignal, p0=(a_guess, T_guess, c_guess, omega_guess, phi0_guess))
    a_fit = popt[0]
    T_fit = popt[1]
    c_fit = popt[2]
    omega_fit = popt[3]
    phi0_fit = popt[4]
    HeterodynSignalFitted = a_fit * np.cos(omega_fit*Time+phi0_fit)* np.exp(-Time/T_fit) + c_fit
    if Output == True:
        print('----FIT RESULTS----')
        print('T = %.2E'% T_fit)
        print('omega = %.2E'% omega_fit)
        plt.figure(figsize = [11,5])
        plt.scatter(Time,HeterodynSignal, s=0.7, color='black');  
        plt.plot(Time, HeterodynSignalFitted, color='dodgerblue');
        plt.grid(which = 'both')
        plt.minorticks_on()
        plt.title(f'{file_name}')
        plt.xlabel("Pulse Length (ns)")
        plt.ylabel('Heterodyne V [ADCU]')
    return T_fit

# List of file names
file_names = [
    '9_5_2024_timedomain_lengthrabi_3dtransmon_square_amp=5000.dat'
]

# Loop through each file
for file_name in file_names:
    Data = np.loadtxt(file_name, delimiter=',')
    RabiFit(Data, Output = True)


#plt.savefig("RabiFit10k",facecolor="w")
