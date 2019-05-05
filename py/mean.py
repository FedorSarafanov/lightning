import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema,argrelmax

a=[1,50,3,4,3,4,3,-20,3,3,3,21,3,3,2,2,3,2]
a=a*15
a=np.array(a)
def moving_average(x, w):
	return np.convolve(x, np.ones(w), 'valid') / w

num=78
data = np.load('../numpydata/data'+str(num)+'.npy')

flux=data[2]
field=data[1]
flux_osci=1

flux_mean=moving_average(flux,3000)
# flux_diff=moving_average(np.diff(flux_mean,n=1),3000)*1000

# for x in np.where(flux_diff>25):
	# print(x)
plt.plot(flux,'black')
# plt.plot(field*0.1,'blue')
# plt.plot(flux_mean, 'red')
# plt.plot(flux_diff,'green')

# x=np.array([0,1,2,3,4,5,4,3,2,1,0])
# # def minmax(y):
# 	# peaks = np.where((y[1:-1] > y[0:-2]) * (y[1:-1] > y[2:]))[0] + 1
# 	# dips = np.where((y[1:-1] < y[0:-2]) * (y[1:-1] < y[2:]))[0] + 1
# 	# return peaks,dips
# for x in argrelmax(flux_diff)[0]:
# 	if flux_diff[x]>30:
# 		plt.plot(x,0,'ro')

plt.show()