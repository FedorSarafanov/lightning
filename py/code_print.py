# avito2rss
import numpy as np
from numpy import array,mean

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt,argrelextrema,hilbert,chirp
from scipy.fftpack import fft
from scipy import stats



# Класс, считывающий и выдающий предварительно обработанные данные
class data_signal(object):

	def __new__(cls, num = 82):
		return super(data_signal, cls).__new__(cls)

	# Считывание и обработка данных
	def __init__(self,num = 82):

		data = np.load('../numpydata/data'+str(num)+'.npy')

		self.raw_flux_data = data[2]
		self.raw_H_x = data[0]
		self.raw_H_y = data[1]
		self.raw_E_z = data[3]

		# рассматривается часовой отрезок, нарезанный на 100 частей
		self.raw_time = np.linspace(0, 3600/100, len(self.raw_flux_data)) 
		self.flux_data = savgol_filter(self.filter50hz(self.raw_flux_data), 1001, 1)


	# Фильтрация 50 Гц
	def filter50hz(self,x):
		b, a = butter(2, 0.050)
		y = filtfilt(b, a, x)
		b, a = butter(5, 0.050/25)
		y = filtfilt(b, a, y)
		return y


plt.figure(1)
ax = plt.subplot(111, projection='polar')
ax.grid(True)


# Выбор обрабатываемых файлов данных, от до (всего есть 0..99)
for i in range(78,82):
	print('Обрабатывается файл ',i)

	obj = data_signal(num = i)

	flux_data = obj.flux_data

	flux_yshift = 20 # сдвиг на графике для обзорного режима

	H_noise_level = 25 #  уровень шумового сигнала (отсчетов)
	flux_diff_interval = 200 # ширина окна дискретной производной
	flux_diff_strike_interval = 8000 # минимальное расстояние между отдельными сериями разрядов (отсчетов)
	flux_diff_noise_level = 2 # уровень шумового сигнала производной (отсчетов)
	flux_padding = 10000 # уширение временного отрезка найденных атмосфериков
	strike_time = 200e-6 # максимальное время разряда 200 мкс (с запасом: можно 100, но вопросы к разрешению)
	re_strike_delta_t=0.06 # 60 мс, интервал между компонентами атмосферика

	t = obj.raw_time+i*36

	Hx=obj.raw_H_x
	Hy=obj.raw_H_y
	Ez=obj.raw_E_z

	Hx = Hx - np.mean(Hx)
	Hy = Hy - np.mean(Hy)
	Ez = Ez - np.mean(Ez)

	plt.figure(2)
	# Построение полей на графике. В масштабе, чтобы сопоставить с данными флюксметра
	plt.plot(t,Hx/50,'red',alpha=0.25)
	plt.plot(t,Hy/50,'green',alpha=0.25)
	plt.plot(t, obj.raw_flux_data+flux_yshift, 'black', alpha=0.2)
	plt.plot(t, obj.flux_data+flux_yshift, 'blue', alpha=0.8)
	# plt.plot(t,Ez/50,'blue',alpha=0.25)

	# Обрезка шумового сигнала
	Hx[np.abs(Hx)<H_noise_level]=0
	Hy[np.abs(Hy)<H_noise_level]=0
	Ez[np.abs(Ez)<H_noise_level]=0


	# Расчет дискретной производной сигнала с флюксметра
	dy=np.zeros(len(flux_data))
	i = 0
	while i<len(flux_data)-flux_diff_interval:
		arr = flux_data[i:i+flux_diff_interval]
		i+=flux_diff_interval
		slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(arr)),arr)
		dy[i:i+flux_diff_interval] = slope*1000


	# Обрезка шумовой производной
	dy[abs(dy)<flux_diff_noise_level]=0



	# Выделение интервалов атмосфериков, с помощью анализа поведения производной
	q = dy.nonzero()
	q = q[0][1:][(q[0][1:]*((np.diff(q)>flux_diff_strike_interval)[0])).nonzero()]
	scopes = np.split(np.linspace(0, len(dy)-1, len(dy)-1,dtype=np.int),q)

	# Функция подсчета числа компонент в атмосферике
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

	# Перебор и анализ атмосфериков
	for i,scope in enumerate(scopes):
		scope = np.delete(scope,np.where(dy[scope]==0))
		if len(scope)!=0:

			# Определение знака разряда и амплитуды скачка квазистатического поля
			padding_scope = np.linspace(scope[0]-flux_padding,scope[-1]+flux_padding,len(scope)+2*flux_padding, dtype=np.int)
			if len(padding_scope)>0:
				scope = padding_scope

			temp = np.argmin(flux_data[scope])-np.argmax(flux_data[scope])
			strike_sign = temp/abs(temp)
			strike_flux_ampl = np.abs(np.max(flux_data[scope])-np.min(flux_data[scope]))

			H_x = Hx[scope[0]:scope[-1]]
			H_y = Hy[scope[0]:scope[-1]]
			E_z = Ez[scope[0]:scope[-1]]

			# Поиск первого разряда
			inds = [np.argmax(np.abs(H_x)), np.argmax(np.abs(H_y)), np.argmax(np.abs(E_z))]
			ampls = [np.max(np.abs(H_x)), np.max(np.abs(H_y)), np.max(np.abs(E_z))]
			indmax = inds[np.argmax(ampls)]
			indmax = np.argmax(np.sqrt(H_x**2+H_y**2))
			p=10

			# plt.axvline(x=t[scope[0]+indmax])
			
			max_scope = np.linspace(scope[0]+indmax-p,scope[0]+indmax+p,p*2,dtype=np.int)
			Hxmax = np.max(Hx[max_scope])
			Hymax = np.max(Hy[max_scope])
			Ezmax = np.max(Ez[max_scope])

			# определение числа компонент
			S = scope[indmax:]
			count = re_strikes_count(t[S],Hy[S])

			
			H = np.sqrt(Hxmax**2+Hymax**2)
			
			# Грубая оценка расстояния до разряда, нормировано по априорным данным
			r = 1/strike_flux_ampl*100*25

			# Определение направления на разряд и построение радарной карты
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

			print('Начало разряда: {} с, амплитуда по флюксметру {}, знак разряда: {}'.format(t[scope[0]],strike_flux_ampl,strike_sign))
			print('Hx={},Hy={}, Ez={}, число компонент {}'.format(Hxmax,Hymax,Ezmax,count))
			print('r={} км, theta={} рад'.format(r, theta))

			# Построение временного графика найденных атмосфериков
			plt.figure(2)
			plt.plot(t[scope],flux_data[scope]+flux_yshift,'red',alpha=1)
			plt.plot(t[scope],Hx[scope]/50,'red',alpha=1)
			plt.plot(t[scope],Hy[scope]/50,'green',alpha=1)
			plt.plot(t[scope],Ez[scope]/50,'blue',alpha=1)

		# Запись атмосферика, с удаленными нулевыми полями (padding)
		scopes[i]=scope

# Вывод осциллограммы и радарной карты
plt.show()