# avito2rss
import numpy as np
from scipy import signal
from numpy import array,mean
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from scipy.fftpack import fft

num=79
data = np.load('../numpydata/data'+str(num)+'.npy')
x=data[2]
t = np.linspace(0, 1.0, len(x))

H_x=data[0]
H_y=data[1]

H_z=data[3]
# print(mean(ch_x),mean(ch2_y,mean(ch_z))



# Now create a lowpass Butterworth filter with a cutoff of 0.059 times the Nyquist rate, 
# or 50 Hz, and apply it to x with filtfilt. The result should be approximately xlow, with no phase shift.
b, a = signal.butter(2, 0.050)
y = signal.filtfilt(b, a, x)
b, a = signal.butter(5, 0.050/25)
y = signal.filtfilt(b, a, y)

th=t
plt.plot(t,x,'black', alpha=0.1)
t,y=t[0::500], y[0::500]

# H_x,H_y,H_z=H_x[0::500],H_y[0::500],H_z[0::500]

# https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
# https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
flux = yhat = savgol_filter(y, 11, 1) # window size 51, polynomial order 3

dy = np.append(np.diff(yhat,n=1),0)
# print(dy)
dy = savgol_filter(dy, 11, 3)
dym=0.05

# plt.plot(t,y,'b',alpha=0.4)
plt.plot(t,yhat,'r',alpha=0.5)
plt.plot(t,dy,'b')


# Хитрофильтрующие алогритмы поиска локальных максимумов и минимумов
def maxima(t,dy):
	indmax=signal.argrelextrema(dy, np.greater)[0]
	# dym = np.median(dy)
	# dym = 0.1
	# plt.plot(t[indmax],dy[indmax],'ro',alpha=0.3)
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

max_ind = M = maxima(t,dy)
min_ind = N = minima(t,dy)



double_ind = []
single_ind = []
tau = 0.04 # коэффициент близости минимума/максимума, подобрать
padding = 20 # коэффициент уширения области разряда для поиска экстремумов

def ampl(flux, inds, padding):
	ind1,ind2 = inds[0], inds[1]
	if ind1-padding<0:
		ind1 = padding
	if ind2+padding>len(flux)-1:
		ind2 = ind2-padding
	arr = flux[(ind1-padding):(ind2+padding)]
	if len(arr)!=0:
		plt.plot(t[(ind1-padding):(ind2+padding)],arr,'blue')
		return abs(np.min(arr)-np.max(arr)), ind1-padding, ind2+padding
	else:
		print(inds, t[(ind1-padding):(ind2+padding)])


for ind in max_ind:
	env = np.array([i for i in min_ind if abs(t[i]-t[ind]) <= tau])
	len_env=len(env)
	if len_env == 0:
		single_ind.append(ind)
	elif len_env == 1:
		amplitude,i1,i2 = ampl(flux,np.sort([ind,env[0]]),padding)
		# double_ind.append([np.sort([ind,env[0]]),amplitude])
		double_ind.append([np.sort(array([i1,i2], dtype=np.int)),amplitude])

positive_ind = []
negative_ind = []

for duplet_ind in double_ind:
	first_ind = int(duplet_ind[0][0])
	last_ind = int(duplet_ind[0][1])
	if dy[first_ind] > dy[last_ind]:
		# print(first_ind, last_ind, dy[first_ind], dy[last_ind])
		negative_ind.append(duplet_ind)
	else:
		positive_ind.append(duplet_ind)

double_ind = np.array(double_ind)
single_ind = np.array(single_ind)
positive_ind = np.array(positive_ind)
negative_ind = np.array(negative_ind)


def spectre(y):
	N=len(y)
	freq = 20000 # 20 kHz
	# N = 1200000 # Number of samplepoints
	T = 1.0 / freq # sample spacing
	# x = np.linspace(0.0, N*T, N)
	# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(480.0 * 2.0*np.pi*x)
	yf = fft(y)
	xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
	plt.figure()
	plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
	plt.show()


# print(t[double_ind])
# print(t[single_ind])
H_xm=mean(H_x[0::500])
H_ym=mean(H_y[0::500])
H_zm=mean(H_z[0::500])
for duplet_ind in negative_ind:
	[start,end] = duplet_ind[0]*500
	Hx=(np.max(H_x[start:end]) if abs(np.max(H_x[start:end]))>abs(np.min(H_x[start:end]))  else np.min(H_x[start:end]))
	Hy=(np.max(H_y[start:end]) if abs(np.max(H_y[start:end]))>abs(np.min(H_y[start:end]))  else np.min(H_y[start:end]))
	Hz=(np.max(H_z[start:end]) if abs(np.max(H_z[start:end]))>abs(np.min(H_z[start:end]))  else np.min(H_z[start:end]))
	print(duplet_ind,Hx,Hy,Hz)
	# spectre(H_x[start:end])
	# print(start,end)
	# print(H_x[start:end])
	plt.plot(th[start:end],H_x[start:end]/30+350,'r')
	plt.plot(th[start:end],H_y[start:end]/30+350,'g')
	plt.plot(th[start:end],H_z[start:end]/30+350,'b')

plt.plot(t[M],dy[M],'ro', markersize=5)
plt.plot(t[N],dy[N],'go', markersize=5)
plt.show()
