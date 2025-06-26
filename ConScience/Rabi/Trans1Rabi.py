import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import Qore as qr

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
    omega_guess = 6.28/(2*np.abs(Time[np.ara
        plt.figure()#figsize = [11,5])
        plt.scatter(Time,HeterodynSignal, s=0.7, color='black', label='Data');  
        plt.plot(Time, HeterodynSignalFitted, color='dodgerblue', label='Fit');
        plt.grid(which = 'both')
        plt.minorticks_on()
        plt.title(f'{FileName}')
        plt.xlabel("Pulse Length (ns)")
        plt.ylabel('Heterodyne V [ADCU]')
        plt.legend()
    return T_fit

FileName = '9_3_2024_timedomain_rabi_transmon1_square_amp=30000.dat'
Data = np.loadtxt(FileName, delimiter=',')

RabiFit(Data, Output = True)

#plt.savefig("RabiFit10k",facecolor="w")
