import numpy as np
from numpy import array,mean
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt,argrelextrema,hilbert,chirp
from scipy.fftpack import fft
from scipy import stats

# Load,  processing, fitting experimental data
class data_signal(object):
	def __new__(cls, num = 82):
		return super(data_signal, cls).__new__(cls)

	# Load and filtering data
	def __init__(self,num = 82):
		data = np.load('../numpydata/data'+str(num)+'.npy')
		self.raw_flux_data = data[2]
		self.raw_H_x = data[0]
		self.raw_H_y = data[1]
		self.raw_E_z = data[3]

		# Length of data range is 1 hour, splits to 100 parts 
		self.raw_time = np.linspace(0, 3600/100, len(self.raw_flux_data)) 
		self.flux_data = savgol_filter(
			self.filter50hz(self.raw_flux_data), 1001, 1)

	# Filter 50 Hz
	def filter50hz(self,x):
		b, a = butter(2, 0.050)
		y = filtfilt(b, a, x)
		return y

plt.figure(1)
ax = plt.subplot(111, projection='polar')
ax.grid(True)

# Select data to work
# for i in range(78,82):
for i in range(80,90):
	print('Processing file ',i)

	obj = data_signal(num = i)

	flux_data = obj.flux_data
	flux_yshift = 20 # Shift flux data for best view
	H_noise_level = 25 #  H-field noise level (counts)
	flux_diff_interval = 200 # Discrete diff window width
	flux_diff_strike_interval = 8000 # minimal length between two strikes
	flux_diff_noise_level = 2 # Diff level noise
	flux_padding = 10000 # Window padding for best quality
	strike_time = 200e-6 # Strike time length (max)
	re_strike_delta_t=0.06 # 60 ms, Strike-series interval
	t = obj.raw_time+i*36

	Hx=obj.raw_H_x
	Hy=obj.raw_H_y
	Ez=obj.raw_E_z

	Hx = Hx - np.mean(Hx)
	Hy = Hy - np.mean(Hy)
	Ez = Ez - np.mean(Ez)

	plt.figure(2)
	# Plot fields
	plt.plot(t,Hx/50,'red',alpha=0.25)
	plt.plot(t,Hy/50,'green',alpha=0.25)
	plt.plot(t, obj.raw_flux_data+flux_yshift, 'black', alpha=0.2)
	plt.plot(t, obj.flux_data+flux_yshift, 'blue', alpha=0.8)
	# plt.plot(t,Ez/50,'blue',alpha=0.25)

	# Cut noise 
	Hx[np.abs(Hx)<H_noise_level]=0
	Hy[np.abs(Hy)<H_noise_level]=0
	Ez[np.abs(Ez)<H_noise_level]=0

	# Discrete diff flux data
	dy=np.zeros(len(flux_data))
	i = 0
	while i<len(flux_data)-flux_diff_interval:
		arr = flux_data[i:i+flux_diff_interval]
		i+=flux_diff_interval
		slope, intercept, r_value, p_value, std_err = stats.linregress(
			np.arange(len(arr)),arr)
		dy[i:i+flux_diff_interval] = slope*1000

	# Cut diff noise level
	dy[abs(dy)<flux_diff_noise_level]=0

	# Search strike series
	q = dy.nonzero()
	q = q[0][1:][(q[0][1:]*((np.diff(q)>flux_diff_strike_interval)[0])).nonzero()]
	scopes = np.split(np.linspace(0, len(dy)-1, len(dy)-1,dtype=np.int),q)

	# Count strike series
	def re_strikes_count(time, field):
		field_min = np.max(field)/6
		time = time[field>field_min]
		count = 1
		for i in range(0,len(time)-1):
			if time[i+1]-time[i]<re_strike_delta_t:
				count+=1
			else:
				break
		return count

	# foreach found strike series
	for i,scope in enumerate(scopes):
		scope = np.delete(scope,np.where(dy[scope]==0))
		if len(scope)!=0:
			try:
				# Def strike sign and amplitude
				padding_scope = np.linspace(scope[0]-flux_padding,scope[-1]+
					flux_padding,len(scope)+2*flux_padding, dtype=np.int)

				if len(padding_scope)>0:
					scope = padding_scope

				temp = np.argmin(flux_data[scope])-np.argmax(flux_data[scope])
				strike_sign = temp/abs(temp)
				strike_flux_ampl = np.abs(np.max(flux_data[scope])-
					np.min(flux_data[scope]))

				H_x = Hx[scope[0]:scope[-1]]
				H_y = Hy[scope[0]:scope[-1]]
				E_z = Ez[scope[0]:scope[-1]]
				# Search first strike component
				if len(H_x)!=0:
					inds = [np.argmax(np.abs(H_x)), np.argmax(np.abs(H_y)), 
						np.argmax(np.abs(E_z))]
					ampls = [np.max(np.abs(H_x)), np.max(np.abs(H_y)), np.max(np.abs(E_z))]
					indmax = inds[np.argmax(ampls)]
					indmax = np.argmax(np.sqrt(H_x**2+H_y**2))
					p=10

					max_scope = np.linspace(scope[0]+indmax-p,scope[0]+indmax+p,p*2,
						dtype=np.int)
					Hxmax = np.max(Hx[max_scope])
					Hymax = np.max(Hy[max_scope])
					Ezmax = np.max(Ez[max_scope])

					# Count strike-series components
					S = scope[indmax:]
					count = re_strikes_count(t[S],Hy[S])
			
					H = np.sqrt(Hxmax**2+Hymax**2)
					
					# Evaluation r from 0 to strike
					r = 1/strike_flux_ampl*100*25

					# Calculate theta angle to strike in polar coord.
					try:
						theta = np.arctan(Hymax/Hxmax)
					except Exception as e:
						pass
					else:
						plt.figure(1)
						if strike_sign>0:
							ax.plot(theta,r,'ro')
						if strike_sign<0:
							ax.plot(theta+np.pi,r,'bo')
					finally:
						pass	
			except Exception as e:
				pass
			else:
				print('strike start: {} s, ampl {}, sign: {}'.format(t[scope[0]],strike_flux_ampl,strike_sign))
				print('Hx={},Hy={}, Ez={}, comp.count {}'.format(Hxmax,Hymax,Ezmax,count))
				print('r={} km, theta={} rad'.format(r, theta))

				# Plot osci. with found strikes
				plt.figure(2)
				plt.plot(t[scope],flux_data[scope]+flux_yshift,'red',alpha=1)
				plt.plot(t[scope],Hx[scope]/50,'red',alpha=1)
				plt.plot(t[scope],Hy[scope]/50,'green',alpha=1)
				plt.plot(t[scope],Ez[scope]/50,'blue',alpha=1)
			finally:
				pass
		# Write modified data withhout zero paddings
		scopes[i]=scope

# Plot all graphs
plt.show()