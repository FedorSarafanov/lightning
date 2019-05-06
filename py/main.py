# avito2rss
import numpy as np
from scipy import signal
from numpy import array,mean
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
from scipy.fftpack import fft
from scipy import stats
from scipy.signal import hilbert, chirp
import time

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

class data_signal(object):

	def __new__(cls):
		# print(path)
		return super(data_signal, cls).__new__(cls)

	def __init__(self):

		num = 90
		data = np.load('../numpydata/data'+str(num)+'.npy')

		self.raw_flux_data = data[2]
		self.raw_H_x = data[0]
		self.raw_H_y = data[1]
		self.raw_H_z = data[3]
		self.raw_time = np.linspace(0, 3600, len(self.raw_flux_data)) # рассматривается часовой отрезок

		self.flux_data = savgol_filter(self.filter50hz(self.raw_flux_data), 1001, 1)

		# xvals = np.linspace(0, len(self.raw_flux_data), len(self.raw_flux_data))
		# x = np.linspace(0, len(self.flux_data), len(self.flux_data))
		# self.flux_data = np.interp(xvals, x, self.flux_data)

	def filter50hz(self,x):
		b, a = signal.butter(2, 0.050)
		y = signal.filtfilt(b, a, x)
		b, a = signal.butter(5, 0.050/25)
		y = signal.filtfilt(b, a, y)
		return y

	def plot(self):
		plt.plot(self.raw_time, self.flux_data)
		plt.plot(self.raw_time, self.raw_flux_data, 'black', alpha=0.3)

	def show(self):
		plt.show()

obj = data_signal()

flux_data = obj.flux_data
count = 200

# for i, E in enumerate(flux_data):
dy=np.ones(len(flux_data))*0
i = 0
while i<len(flux_data)-count:
	arr = flux_data[i:i+count]
	i+=count
	slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(arr)),arr)
	dy[i:i+count] = slope*1000

plt.plot(obj.raw_time,obj.raw_time*0, 'black', alpha=0.1)

for i, dE in enumerate(dy):
	if abs(dE)<2:
		dy[i]=0
	pass
tic()
q = dy.nonzero()
q = q[0][1:][(q[0][1:]*((np.diff(q)>8000)[0])).nonzero()]

scopes = np.split(np.linspace(0, len(dy)-1, len(dy)-1,dtype=np.int),q)
t = obj.raw_time

for scope in scopes:
	scope = np.delete(scope,np.where(dy[scope]==0))
	if len(scope)!=0:
		mini = scope[np.argmin(dy[scope])]
		maxi = scope[np.argmax(dy[scope])]
		plt.plot(t[scope],dy[scope])
		plt.plot(t[mini],dy[mini],'ro')
		plt.plot(t[maxi],dy[maxi],'go')
toc()

# plt.plot(obj.raw_time, dy)

obj.plot()
obj.show()