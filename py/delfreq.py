import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema

num=79
data = np.load('../numpydata/data'+str(num)+'.npy')
x=data[2]
t = np.linspace(0, 1.0, len(x))


def scroll(y):
	z=0*y
	for i in range(0,len(z)):
		pass
	return z


# Now create a lowpass Butterworth filter with a cutoff of 0.059 times the Nyquist rate, 
# or 50 Hz, and apply it to x with filtfilt. The result should be approximately xlow, with no phase shift.
b, a = signal.butter(2, 0.050)
y = signal.filtfilt(b, a, x)
b, a = signal.butter(5, 0.050/25)
y = signal.filtfilt(b, a, y)

plt.plot(t,x,'black')
t,y=t[0::500], y[0::500]


# https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
# https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
yhat = savgol_filter(y, 11, 1) # window size 51, polynomial order 3

dy = np.append(np.diff(yhat,n=1),0)
dym=0.05

# plt.plot(t,y,'b',alpha=0.4)
plt.plot(t,yhat,'r')
plt.plot(t,dy,'b')


# Хитрофильтрующие алогритмы поиска локальных максимумов и минимумов
def maxima(t,dy):
	indmax=signal.argrelextrema(dy, np.greater)[0]
	# dym = np.median(dy)
	# dym = 0.1

	extrema=np.array([],dtype=np.int)
	for i in indmax:
		if dy[i]>np.abs(7*dym):
			if extrema.size>0:
				if abs(t[extrema[-1]]-t[i])>0.05:
					extrema=np.append(extrema,i)
					pass
			else:
				pass
				extrema=np.append(extrema,i)
	return extrema

def minima(t,dy):
	indmax=signal.argrelextrema(dy, np.less)[0]
	# dym = 0.1

	extrema=np.array([],dtype=np.int)
	for i in indmax:
		if dy[i]<-np.abs(7*dym):
			if extrema.size>0:
				if abs(t[extrema[-1]]-t[i])>0.05:
					extrema=np.append(extrema,i)
					pass
			else:
				pass
				extrema=np.append(extrema,i)
	return extrema

M=maxima(t,dy)
N=minima(t,dy)
plt.plot(t[M],dy[M],'ro', markersize=5)
plt.plot(t[N],dy[N],'go', markersize=5)
plt.show()
