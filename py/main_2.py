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
from matplotlib import rc

rc('font',**{'family':'serif'})
rc('text', usetex=True)
rc('text.latex',unicode=True)
rc('text.latex',preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex',preamble=r'\usepackage[russian]{babel}')
plt.rcParams.update({'font.size': 14})
# (?# plt.xlabel(r'\textbf{Время, с}'))
# plt.yticks([])

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

		num = 79
		data = np.load('../numpydata/data'+str(num)+'.npy')

		H_noise_level = 25 # уровень шума поля H
		self.flux_diff_interval = 200 # количество отсчетов, усредняемых при расчете производной
		self.flux_diff_stroke_interval = 8000 # расстояние между крупными разрядами, отсчетов
		self.flux_diff_noise_level = 2 # шум производной
		self.flux_padding = 10000 # поля найденного пика
		self.stroke_time = 200e-6 # 200 мкс (с запасом: можно 100, но вопросы к разрешению)
		self.stroke_padding = 10 # отсчетов, область поиска разрешенных компонент

		self.raw_flux_data = data[2]
		self.Hx = data[0] - np.mean(data[0])
		self.Hy = data[1] - np.mean(data[1])
		self.Ez = data[3] - np.mean(data[3])

		self.Hx[np.abs(self.Hx)<H_noise_level]=0
		self.Hy[np.abs(self.Hy)<H_noise_level]=0
		self.Ez[np.abs(self.Ez)<H_noise_level]=0

		self.t = np.linspace(0, 3600/100, len(self.raw_flux_data)) # рассматривается 1/100 часа

		self.flux_data = savgol_filter(self.filter50hz(self.raw_flux_data), 1001, 1)

	# def 

	def filter50hz(self,x):
		b, a = signal.butter(2, 0.050)
		y = signal.filtfilt(b, a, x)
		b, a = signal.butter(5, 0.050/25)
		y = signal.filtfilt(b, a, y)
		return y

	def plot(self):
		plt.plot(self.raw_time, self.flux_data, 'blue', alpha=0.5)
		plt.plot(self.raw_time, self.raw_flux_data, 'black', alpha=0.2)

	def show(self):
		plt.show()

tic()
obj = data_signal()



# for i, E in enumerate(flux_data):
dy=np.zeros(len(flux_data))
i = 0
while i<len(flux_data)-flux_diff_interval:
	arr = flux_data[i:i+flux_diff_interval]
	i+=flux_diff_interval
	slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(arr)),arr)
	dy[i:i+flux_diff_interval] = slope*1000

plt.plot(obj.raw_time,obj.raw_time*0, 'black', alpha=0.1)

for i, dE in enumerate(dy):
	if abs(dE)<flux_diff_noise_level:
		dy[i]=0
	pass



q = dy.nonzero()
q = q[0][1:][(q[0][1:]*((np.diff(q)>flux_diff_stroke_interval)[0])).nonzero()]

scopes = np.split(np.linspace(0, len(dy)-1, len(dy)-1,dtype=np.int),q)
t = obj.raw_time


for i,scope in enumerate(scopes):
	scope = np.delete(scope,np.where(dy[scope]==0))
	if len(scope)!=0:
		# mini = scope[np.argmin(dy[scope])]
		# maxi = scope[np.argmax(dy[scope])]
		# plt.plot(t[mini],dy[mini],'ro')
		# plt.plot(t[maxi],dy[maxi],'go')
		# plt.plot(t[scope],dy[scope])

		padding_scope = np.linspace(scope[0]-flux_padding,scope[-1]+flux_padding,len(scope)+2*flux_padding, dtype=np.int)
		if len(padding_scope)>0:
			scope = padding_scope

		temp = np.argmin(flux_data[scope])-np.argmax(flux_data[scope])
		stroke_sign = temp/abs(temp)
		stroke_flux_ampl = np.abs(np.max(flux_data[scope])-np.min(flux_data[scope]))

		H_x = Hx[scope[0]:scope[-1]]
		H_y = Hy[scope[0]:scope[-1]]
		E_z = Ez[scope[0]:scope[-1]]

		# Поиск первого разряда
		inds = [np.argmax(np.abs(H_x)), np.argmax(np.abs(H_y)), np.argmax(np.abs(E_z))]
		ampls = [np.max(np.abs(H_x)), np.max(np.abs(H_y)), np.max(np.abs(E_z))]
		indmax = inds[np.argmax(ampls)]
		indmax = np.argmax(np.sqrt(H_x**2+H_y**2))
		# print(scope[0])
		plt.axvline(x=t[scope[0]+indmax])
		# print(indmax)
		# print(H_x[indmax],Hx[scope[0]+indmax])
		max_scope = np.linspace(scope[0]+indmax-stroke_padding,scope[0]+indmax+stroke_padding,stroke_padding*2,dtype=np.int)
		# print(max_scope)
		Hxmax = np.max(Hx[max_scope])
		Hymax = np.max(Hy[max_scope])
		Ezmax = np.max(Ez[max_scope])
		# print(Hxmax,Hymax,Ezmax)

		# определение числа компоненты
		maxs = argrelextrema(Hx[scope], np.greater)[0]
		# print(maxs)
		count = 1
		for j in range(1,len(maxs)):
			if Hx[scope[maxs[j]]]>500:
				if t[scope[maxs[j]]]-t[scope[maxs[j-1]]]<0.06:
					count +=1
				# print(t[scope[maxs[j]]])

		print('Начало разряда: {} с, амплитуда по флюксметру {}, знак разряда: {}'.format(t[scope[0]],stroke_flux_ampl,stroke_sign))
		# print('Hx={},Hy={}, Ez={}, число компонент {}'.format(Hxmax,Hymax,Ezmax,count))
		print('Hx={},Hy={}, Ez={}'.format(Hxmax,Hymax,Ezmax))
		plt.plot(t[scope],flux_data[scope],'red',alpha=1)
		plt.plot(t[scope],Hx[scope]/50,'red',alpha=1)
		plt.plot(t[scope],Hy[scope]/50,'green',alpha=1)
		plt.plot(t[scope],Ez[scope]/50,'blue',alpha=1)
	scopes[i]=scope
toc()

plt.figure()
ax = plt.subplot(111, projection='polar')
ax.plot([], [])
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()

# plt.plot(obj.raw_time, dy)

# obj.plot()
# obj.show()