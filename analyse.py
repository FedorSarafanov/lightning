import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter#, butter, filtfilt,argrelextrema,hilbert,chirp

data = np.loadtxt('all_str_2.dat', dtype=np.float32, skiprows=1, usecols=(0,1,2,3,4,5,6,7,8))
data = data[data[:,0].argsort()]
time = data.T[0]
flux = data.T[1]
sign = data.T[2]
hx = data.T[3]
hy = data.T[4]
ez = data.T[5]
count = data.T[6]
r = data.T[7]
theta = data.T[8]

flux=flux/150*300
flux = savgol_filter(flux, 31, 1)
r = savgol_filter(r, 101, 1)

plt.figure()
# plt.plot(time,flux)
plt.ylabel(r'$E$,  $\dfrac{в}{м}$')
plt.xlabel(r'$t$,  с')
# plt.plot(time,np.exp(time/560))
# plt.plot(time,np.exp(time/625)+10)
ind=np.where(np.abs(ez)/np.sqrt(hx**2+hy**2)>0)
R=np.abs(ez)/np.sqrt(hx**2+hy**2)
Ri = R[ind]
Ri = savgol_filter(Ri, 3, 1)
plt.ylabel(r'$\dfrac{E_z}{H_φ}$')
plt.xlabel(r'$t$,  с')
# plt.plot(time[ind],Ri)

p, x = np.histogram(hx**2+hy**2, bins=100)
x = x[:-1] + (x[1] - x[0])/2
# print(p,x)
plt.plot(x,p)
plt.plot(-x,p)
# plt.hist(p,bins=30)
# plt.figure()
# plt.plot(time,r)
plt.show()
