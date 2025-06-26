import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import sys
import os
import matplotlib

dataE40 = np.loadtxt('E40.csv', delimiter=',')
dataE0 = np.loadtxt('E0.csv', delimiter=',')

xE0data = (dataE0[:,0])
yE0data = (dataE0[:,1])
xE40data = (dataE40[:,0])
yE40data = (dataE40[:,1])
xE0data = np.array(xE0data)
yE0data = np.array(yE0data)

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

plt.figure(figsize = [11,5])
plt.plot(xE40data, yE40data + 40, color="orange", label="P1=40dB   P2=0dB")
plt.plot(xE0data, yE0data, label="P1=0dB P2=0dB")  
plt.grid(which = 'both')
plt.ticklabel_format(useOffset=False)
plt.title("Etched Cavity Power Dependence T=10mK")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Insertion Loss (dB)")
plt.yscale('linear')
plt.ylim(-100, 10)
plt.axvline(7.7743906, color="black", linestyle="dashed", label=r'$f_0$=7.7743906 GHz')
plt.legend(loc='upper left');

#plt.savefig("Etched Cavity Power Dependence T=10mK",facecolor="w")


dataQCWL = np.loadtxt('WireLength.csv', delimiter=',')

xdata = (dataQCWL[:,0])
ydata = (dataQCWL[:,1])

# calculate polynomial
z = np.polyfit(xdata, ydata, 6)
f = np.poly1d(z)

# calculate new x's and y's
x_new = np.linspace(xdata[0], xdata[-1], 50)
y_new = f(x_new)

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

plt.figure(figsize = [11,5])
plt.scatter(xdata,ydata,label="Data") 
plt.plot(x_new, y_new, color="darkorchid",label="Fit")
plt.grid(which = 'both')
plt.ticklabel_format(useOffset=False)
plt.title(r"$\mathbf{Coupling\,Quality\,vs.\,Wire\,Length}$")
plt.xlabel(r"$\mathbf{Wire\,Length\,(mm)}$")
plt.ylabel(r"$\mathbf{Coupling\,Quality\,Factor}$")
plt.yscale('linear')
plt.legend(loc='upper right');

#plt.savefig("Coupling Quality vs Wire Length",facecolor="w")
