import numpy as np
from matplotlib import pyplot as plt 
import scipy.fftpack

i=10
data=np.load('../numpydata/data'+str(i)+'.npy')
y=data[2]
N=len(y)


freq = 20000 # 20 kHz
# N = 1200000 # Number of samplepoints
T = 1.0 / freq # sample spacing

# x = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(480.0 * 2.0*np.pi*x)
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))

fig, ax = plt.subplots()
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()