#Authors:
#Francesco Vischi

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scqubits as scq

####### Resonance on transmission (cavity 3D)
def Resonance3DCavity(fr, f0, Qt, Qe):
    S21_Power = 10*np.log10(Qe**-2/(Qt**-2+4*(fr/f0-1)**2))
    return S21_Power

def PlotResonance3DCavity(f0, Qi, Qe, NFWHM=5):
    Qt = (1/Qi+1/Qe)**-1
    fr = np.linspace(f0-NFWHM*f0/Qt/2,f0+NFWHM*f0/Qt/2,1000)
    S21 = Resonance3DCavity(fr, f0, Qt, Qe)
    plt.figure(figsize = [11,5])
    plt.plot(fr,S21)
    plt.grid(which = 'both')
    plt.minorticks_on()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('|S21|^2 [dB]')

def Resonator3DCavityFit(fr, S21_dB, PwrRange = 20):
    peak = np.max(S21_dB)
    fr_peak = fr[np.argmax(S21_dB)]
    fr_3dB = 2*np.abs(fr[np.argmin(np.abs(S21_dB-peak+3.01))]-fr_peak) #FWHM
    Qt = fr_peak/fr_3dB
    Qe = Qt/np.sqrt(10**(peak/10))
    print('----PRE-FIT PARAMETERS ESTIMATION----')
    print('f0[Hz] = %.6E'% fr_peak)
    print('FWHM [Hz] = %.3E'% fr_3dB)
    print('Qt = %.3E'% Qt)
    print('IL[dB] = %.2f' % peak)
    print('Qe = %.3E'% Qe)
    #windowing
    peak_idx = np.argmax(S21_dB)
    Idx_width = np.abs(peak_idx-np.argmin(np.abs(S21_dB-peak+PwrRange)))
    fr_wndw = fr[peak_idx-Idx_width:peak_idx+Idx_width]
    S21_wndw = S21_dB[peak_idx-Idx_width:peak_idx+Idx_width]
    #done windowing
    Initial = [fr_peak, Qt,Qe]
    popt, pcov = curve_fit(Resonance3DCavity, fr_wndw, S21_wndw, p0=Initial)
    f0_fit = popt[0]
    Qt_fit = popt[1]
    Qe_fit = popt[2]
    Qi_fit = (1/Qt_fit-1/Qe_fit)**-1
    print('----FIT RESULTS----')
    print('f0[Hz] = %.6E'% f0_fit)
    print('Qt = %.3E'% Qt_fit)
    print('Qe = %.3E' % Qe_fit)
    print('Qi = %.3E' % Qi_fit)
    plt.figure(figsize = [11,5])
    plt.plot(fr,S21_dB, label = 'data')
    plt.plot(fr_wndw, Resonance3DCavity(fr_wndw,f0_fit,Qt_fit,Qe_fit), label='fit')
    plt.grid(which = 'both')
    plt.minorticks_on()
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('|S21|^2 [dB]')
    return f0_fit,Qt_fit,Qe_fit,Qi_fit

def Resonator3DCavityFitSilent(fr, S21_dB, PwrRange = 20):
    peak = np.max(S21_dB)
    fr_peak = fr[np.argmax(S21_dB)]
    fr_3dB = 2*np.abs(fr[np.argmin(np.abs(S21_dB-peak+3.01))]-fr_peak) #FWHM
    Qt = fr_peak/fr_3dB
    Qe = Qt/np.sqrt(10**(peak/10))
    #windowing
    peak_idx = np.argmax(S21_dB)
    Idx_width = np.abs(peak_idx-np.argmin(np.abs(S21_dB-peak+PwrRange)))
    fr_wndw = fr[peak_idx-Idx_width:peak_idx+Idx_width]
    S21_wndw = S21_dB[peak_idx-Idx_width:peak_idx+Idx_width]
    #done windowing
    Initial = [fr_peak, Qt,Qe]
    popt, pcov = curve_fit(Resonance3DCavity, fr_wndw, S21_wndw, p0=Initial)
    f0_fit = popt[0]
    Qt_fit = popt[1]
    Qe_fit = popt[2]
    Qi_fit = (1/Qt_fit-1/Qe_fit)**-1
    return f0_fit,Qt_fit,Qe_fit,Qi_fit

####### Resonance coupled to feedline (hanger resonator)
def ResonatorHanger(fr, f0, Qt, Qe, A):
    S21power = (1-Qt/Qe/(1+4*Qt**2*(fr/f0-1)**2))**2 + (2*Qt**2*(fr/f0-1)/Qe/(1+4*Qt**2*(fr/f0-1)**2))**2
    S21power = A + 10*np.log10(S21power)
    return S21power

def ResonatorHangerAsymmetric(fr, f0, Qt, Qe, A, df):
    ResLinear = 1 - (1/Qe - 2*1j*df/f0)/(1/Qt+2*1j*(fr/f0-1))
    S21power = A + 20*np.log10(abs(ResLinear));
    return S21power
    
def ResonatorHangerFitSilent(fr, S21_dB, PwrRange = None):
    peak = np.min(S21_dB)
    fr_peak = fr[np.argmin(S21_dB)]
    Prefactor = (S21_dB[0]+S21_dB[-1])/2
    fr_3dB = 2*np.abs(fr[np.argmin(np.abs(S21_dB-peak-3.01))]-fr_peak) #FWHM
    Qi = fr_peak/fr_3dB
    QtOverQe = 1- 10**((peak-Prefactor)/20)
    Qe = Qi*(1/QtOverQe-1)
    Qt = (1/Qe + 1/Qi)**-1
    #windowing
    if PwrRange == None :
        fr_wndw = fr
        S21_wndw = S21_dB
    else:
        peak_idx = np.argmin(S21_dB)
        Idx_width = np.abs(peak_idx-np.argmin(np.abs(S21_dB-Prefactor+PwrRange)))
        fr_wndw = fr[peak_idx-Idx_width:peak_idx+Idx_width]
        S21_wndw = S21_dB[peak_idx-Idx_width:peak_idx+Idx_width]
    #done windowing
    Initial = [fr_peak, Qt,Qe, Prefactor]
    popt, pcov = curve_fit(ResonatorHanger, fr_wndw, S21_wndw, p0=Initial)
    f0_fit = popt[0]
    Qt_fit = popt[1]
    Qe_fit = popt[2]
    A_fit = popt[3]
    Qi_fit = (1/Qt_fit-1/Qe_fit)**-1
    return f0_fit,Qt_fit,Qe_fit,Qi_fit, A_fit

def ResonatorHangerVsPower(S21_dBDataMatrix, PwrRange = None):
    fr = S21_dBDataMatrix[0,1:]
    Pwrs = S21_dBDataMatrix[1:,0]
    S21_dB_Matrix = np.transpose(S21_dBDataMatrix[1:,1:])
    plt.figure(figsize = [11,5])
    for c1 in range(np.size(Pwrs)):
        plt.scatter(fr,S21_dB_Matrix[:,c1], s=0.7, label= str(Pwrs[c1]))  
    plt.grid(which = 'both')
    plt.minorticks_on()
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('|S21|^2 [dB]')
    f0_fit = []
    Qi_fit = []
    Qe_fit = []
    Qt_fit = []
    A_fit = []
    print('||  Att[dB]  ||     f0[Hz]    ||      Qt    ||     Qe     ||     Qi     ||   A[dB]   ')
    for c1 in range(np.size(Pwrs)):
        S21_dB = S21_dB_Matrix[:,c1]
        tmp1, tmp2, tmp3, tmp4, tmp5 = ResonatorHangerFitSilent(fr, S21_dB, PwrRange)
        print('||   %3d   '% Pwrs[c1], ' || %.6E'% tmp1,  ' || %.3E'% tmp2,' || %.3E' % tmp3,' || %.3E' % tmp4,' || %.3E' % tmp5)
        f0_fit.append(tmp1)
        Qt_fit.append(tmp2)
        Qe_fit.append(tmp3)
        Qi_fit.append(tmp4)
        A_fit.append(tmp5)
    return f0_fit,Qt_fit,Qe_fit,Qi_fit, A_fit

def ResonatorHangerFit(fr, S21_dB, PwrRange = None):
    peak = np.min(S21_dB)
    fr_peak = fr[np.argmin(S21_dB)]
    Prefactor = (S21_dB[0]+S21_dB[-1])/2
    fr_3dB = 2*np.abs(fr[np.argmin(np.abs(S21_dB-peak-3.01))]-fr_peak) #FWHM
    Qi = fr_peak/fr_3dB
    QtOverQe = 1- 10**((peak-Prefactor)/20)
    Qe = Qi*(1/QtOverQe-1)
    Qt = (1/Qe + 1/Qi)**-1
    print('----PRE-FIT PARAMETERS ESTIMATION----')
    print('f0[Hz] = %.6E'% fr_peak)
    print('FWHM [Hz] = %.3E'% fr_3dB)
    print('Qt = %.3E'% Qt)
    print('IL[dB] = %.2f' % peak)
    print('Qe = %.3E'% Qe)
    print('A = %.3E'% Prefactor)
    #windowing
    if PwrRange == None :
        fr_wndw = fr
        S21_wndw = S21_dB
    else:
        peak_idx = np.argmin(S21_dB)
        Idx_width = np.abs(peak_idx-np.argmin(np.abs(S21_dB-Prefactor+PwrRange)))
        fr_wndw = fr[peak_idx-Idx_width:peak_idx+Idx_width]
        S21_wndw = S21_dB[peak_idx-Idx_width:peak_idx+Idx_width]
    #done windowing
    Initial = [fr_peak, Qt,Qe, Prefactor]
    popt, pcov = curve_fit(ResonatorHanger, fr_wndw, S21_wndw, p0=Initial, bounds = (np.array([0,0,0,-np.inf]),np.array([np.inf,np.inf,np.inf,np.inf])))
    f0_fit = popt[0]
    Qt_fit = popt[1]
    Qe_fit = popt[2]
    A_fit = popt[3]
    Qi_fit = (1/Qt_fit-1/Qe_fit)**-1
    print('----FIT RESULTS----')
    print('f0[Hz] = %.6E'% f0_fit)
    print('Qt = %.3E'% Qt_fit)
    print('Qe = %.3E' % Qe_fit)
    print('Qi = %.3E' % Qi_fit)
    print('A = %.3E' % A_fit)
    plt.figure(figsize = [11,5])
    plt.plot(fr,S21_dB, label = 'data')
    plt.plot(fr_wndw, ResonatorHanger(fr_wndw,f0_fit,Qt_fit,Qe_fit, A_fit), label='fit')
    plt.grid(which = 'both')
    plt.minorticks_on()
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('|S21|^2 [dB]')
    return f0_fit,Qt_fit,Qe_fit,Qi_fit, A_fit

def ResonatorHangerAsymmetricFit(fr, S21_dB, PwrRange = None, VerboseOutput = True):
    peak = np.min(S21_dB)
    fr_peak = fr[np.argmin(S21_dB)]
    Prefactor = (S21_dB[0]+S21_dB[-1])/2
    fr_3dB = 2*np.abs(fr[np.argmin(np.abs(S21_dB-peak-3.01))]-fr_peak) #FWHM
    Qi = fr_peak/fr_3dB
    QtOverQe = 1- 10**((peak-Prefactor)/20)
    Qe = Qi*(1/QtOverQe-1)
    Qt = (1/Qe + 1/Qi)**-1
    if VerboseOutput:
        print('----PRE-FIT PARAMETERS ESTIMATION----')
        print('f0[Hz] = %.6E'% fr_peak)
        print('FWHM [Hz] = %.3E'% fr_3dB)
        print('Qt = %.3E'% Qt)
        print('IL[dB] = %.2f' % peak)
        print('Qe = %.3E'% Qe)
        print('A = %.3E'% Prefactor)
    #windowing
    if PwrRange == None :
        fr_wndw = fr
        S21_wndw = S21_dB
    else:
        peak_idx = np.argmin(S21_dB)
        Idx_width = np.abs(peak_idx-np.argmin(np.abs(S21_dB-Prefactor+PwrRange)))
        fr_wndw = fr[peak_idx-Idx_width:peak_idx+Idx_width]
        S21_wndw = S21_dB[peak_idx-Idx_width:peak_idx+Idx_width]
    #done windowing
    Initial = [fr_peak, Qt,Qe, Prefactor, 0]
    popt, pcov = curve_fit(ResonatorHangerAsymmetric, fr_wndw, S21_wndw, p0=Initial, bounds = (np.array([0,0,0,-np.inf,-np.inf]),np.array([np.inf,np.inf,np.inf,np.inf,np.inf])))
    f0_fit = popt[0]
    Qt_fit = popt[1]
    Qe_fit = popt[2]
    A_fit = popt[3]
    df_fit = popt[4]
    Qi_fit = (1/Qt_fit-1/Qe_fit)**-1
    if VerboseOutput:
        print('----FIT RESULTS----')
        print('f0[Hz] = %.6E'% f0_fit)
        print('Qt = %.3E'% Qt_fit)
        print('Qe = %.3E' % Qe_fit)
        print('Qi = %.3E' % Qi_fit)
        print('A = %.3E' % A_fit)
        print('Deltaf = %.3E' % df_fit)
        plt.figure(figsize = [11,5])
        plt.scatter(fr,S21_dB, label = 'data', s=0.7, )
        plt.plot(fr_wndw, ResonatorHangerAsymmetric(fr_wndw,f0_fit,Qt_fit,Qe_fit, A_fit,df_fit), label='fit', color= 'r')
        plt.grid(which = 'both')
        plt.minorticks_on()
        plt.legend()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('|S21|^2 [dB]')
    return f0_fit,Qt_fit,Qe_fit,Qi_fit, A_fit, df_fit

def ResonatorHangerAsymmetricFitvsPower(S21_dBDataMatrix, PwrRange = None):
    fr = S21_dBDataMatrix[0,1:]
    Pwrs = S21_dBDataMatrix[1:,0]
    S21_dB_Matrix = np.transpose(S21_dBDataMatrix[1:,1:])
    plt.figure(figsize = [11,5])
    for c1 in range(np.size(Pwrs)):
        plt.scatter(fr,S21_dB_Matrix[:,c1], s=0.7, label= str(Pwrs[c1]))  
    plt.grid(which = 'both')
    plt.minorticks_on()
    plt.legend()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('|S21|^2 [dB]')
    f0_fit = []
    Qi_fit = []
    Qe_fit = []
    Qt_fit = []
    A_fit = []
    df_fit = []
    print('||  Att[dB]  ||     f0[Hz]    ||      Qt    ||     Qe     ||     Qi     ||   A[dB]     || Deltaf[Hz]')
    for c1 in range(np.size(Pwrs)):
        S21_dB = S21_dB_Matrix[:,c1]
        tmp1, tmp2, tmp3, tmp4, tmp5, tmp6 = ResonatorHangerAsymmetricFit(fr, S21_dB, PwrRange, VerboseOutput = False)
        print('||   %3d   '% Pwrs[c1], ' || %.6E'% tmp1,  ' || %.3E'% tmp2,' || %.3E' % tmp3,' || %.3E' % tmp4,' || %.3E' % tmp5,' || %.3E' % tmp6)
        f0_fit.append(tmp1)
        Qt_fit.append(tmp2)
        Qe_fit.append(tmp3)
        Qi_fit.append(tmp4)
        A_fit.append(tmp5)
        df_fit.append(tmp6)
    return f0_fit,Qt_fit,Qe_fit,Qi_fit, A_fit, df_fit

## Qubit dispersive shift vs flux
def ResonatorHangerDispersiveVsFluxFit(S21vsFluxvsPow_dB, PwrRange = None):
    Nflux = np.shape(S21vsFluxvsPow_dB)[0]
    fr = S21vsFluxvsPow_dB[0,1:]
    flux = S21vsFluxvsPow_dB[1:,0]
    f0_vec = []
    Qe_vec = []
    Qi_vec = []
    A_vec = []
    for c1 in range(1,Nflux):
        tmp = ResonatorHangerFitSilent(fr, S21vsFluxvsPow_dB[c1,1:], PwrRange)
        f0_vec.append(tmp[0])
        Qe_vec.append(tmp[1])
        Qi_vec.append(tmp[2])
        A_vec.append(tmp[3])
    f0_vec = np.array(f0_vec)
    Qe_vec = np.array(Qe_vec)
    Qi_vec = np.array(Qi_vec)
    A_vec = np.array(A_vec)
    plt.figure(figsize = [11,5])
    plt.plot(flux, f0_vec, label = 'Dressed f0')
    plt.grid(which = 'both')
    plt.minorticks_on()
    plt.xlabel('Flux [measurement units]')
    plt.ylabel('Resonance [Hz]')
    plt.legend()
    return flux, f0_vec, Qe_vec, Qi_vec, A_vec

# T1 fit
def T1Fit(Data, Output = True):
    Time = Data[:,0]
    HeterodynSignal = Data[:,1]
    a_guess = HeterodynSignal[1]-HeterodynSignal[-1]
    T1_guess = Time[-1]/5
    c_guess = HeterodynSignal[-1]
    popt, pcov = curve_fit(lambda t, a, T1, c: a * np.exp(-T1 * t) + c, Time, HeterodynSignal, p0=(a_guess, T1_guess, c_guess))
    a = popt[0]
    T1 = popt[1]
    c = popt[2]
    HeterodynSignalFitted = a * np.exp(-T1 * Time) + c
    if Output == True:
        print('----FIT RESULTS----')
        print('T1 = %.2E'% T1)
        plt.figure(figsize = [11,5])
        plt.scatter(Time,HeterodynSignal, s=0.7)  
        plt.plot(Time, HeterodynSignalFitted, 'r')
        plt.grid(which = 'both')
        plt.minorticks_on()
        plt.xlabel('Time')
        plt.ylabel('Heterodyne V [ADCU]')
    return T1

# Energy decay fit
def EnergyDecayFit(Data, Output = True):
    Time = Data[:,0]
    HeterodynSignal = Data[:,1]
    a_guess = HeterodynSignal[1]-HeterodynSignal[-1]
    T_guess = Time[-1]/5
    c_guess = HeterodynSignal[-1]
    popt, pcov = curve_fit(lambda t, a, T, c: a * np.exp(-t/T) + c, Time, HeterodynSignal, p0=(a_guess, T_guess, c_guess))
    a_fit = popt[0]
    T_fit = popt[1]
    c_fit = popt[2]
    HeterodynSignalFitted = a_fit * np.exp(-Time/T_fit) + c_fit
    if Output == True:
        print('----FIT RESULTS----')
        print('T = %.2E'% T_fit)
        plt.figure(figsize = [11,5])
        plt.scatter(Time,HeterodynSignal, s=0.7)  
        plt.plot(Time, HeterodynSignalFitted, 'r')
        plt.grid(which = 'both')
        plt.minorticks_on()
        plt.xlabel('Time')
        plt.ylabel('Heterodyne V [ADCU]')
    return T_fit

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
    phi0_fit = popt[3]
    HeterodynSignalFitted = a_fit * np.cos(omega_fit*Time+phi0_fit)* np.exp(-Time/T_fit) + c_fit
    if Output == True:
        print('----FIT RESULTS----')
        print('T = %.2E'% T_fit)
        plt.figure(figsize = [11,5])
        plt.scatter(Time,HeterodynSignal, s=0.7)  
        plt.plot(Time, HeterodynSignalFitted, 'r')
        plt.grid(which = 'both')
        plt.minorticks_on()
        plt.xlabel('Time')
        plt.ylabel('Heterodyne V [ADCU]')
    return T_fit

####### Transmon EJ EC fit
def TransmonFitting(f01_data, alpha_data, Acc = 1e-3, Verbose = True):
    ECseed = alpha_data
    EJseed = f01_data**2/(8*ECseed)
    Prec = 0.5
    chi2_ = 1
    while chi2_>Acc :
        Nmesh = 41
        if Verbose == True:
            print('X', end = '')
        EC_vec = np.linspace(ECseed*(1-Prec),ECseed*(1+Prec),Nmesh)
        EJ_vec = np.linspace(EJseed*(1-Prec),EJseed*(1+Prec),Nmesh)
        chi2 = np.zeros((Nmesh,Nmesh))
        MyTr = scq.Transmon(EJ = EJseed, EC= ECseed, ng=0, ncut = 31)
        for c1 in range(0,Nmesh):
            for c2 in range(0,Nmesh):
                MyTr.EJ = EJ_vec[c1]
                MyTr.EC = EC_vec[c2]
                Eigs = MyTr.eigenvals()
                f01 = Eigs[1]-Eigs[0]
                alpha = (Eigs[1]-Eigs[0]) - (Eigs[2]-Eigs[1])
                chi2[c1,c2] = np.sqrt((f01-f01_data)**2+(alpha-alpha_data)**2)    
        MinimumPosition = np.unravel_index(np.argmin(chi2, axis=None), chi2.shape)
        chi2_ = chi2[MinimumPosition[0],MinimumPosition[1]]
        ECseed = EC_vec[MinimumPosition[1]]
        EJseed = EJ_vec[MinimumPosition[0]]
        if MinimumPosition[0] == 0 or MinimumPosition[1] == 0 or MinimumPosition[0] == Nmesh or MinimumPosition[1] == Nmesh:
            Prec = Prec
        else:
            Prec = Prec/3
    if Verbose == True:
        print(' ')
        print('EJ = %f'%(EJseed))
        print('EC = %f'%(ECseed))
        print('f01 (numeric) = %f'%(f01))
        print('alpha (numeric) = %f'%(alpha))
        print('f02 (numeric) = %f'%(Eigs[2]-Eigs[0]))
    return EJseed, ECseed, f01, alpha, chi2_

