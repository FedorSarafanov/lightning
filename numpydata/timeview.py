import numpy as np
from matplotlib import pyplot as plt 
import scipy.fftpack
import sys

# i=10
# data=np.load('../numpydata/data'+str(i)+'.npy')
# y=data[2]
ch=2
temp=np.load('../numpydata/fastview.npy')[ch]
lims=(np.min(temp),np.max(temp))

def load_data(i):
	global lims
	if ax.get_xlabel()!=str(i):
		if i=='fast':
			data=np.load('../numpydata/fastview.npy')
		else:
			data=np.load('../numpydata/data'+str(i)+'.npy')
		y=data[ch]
		line.set_data(np.linspace(0,1,len(y)),y)
		# ax.set_ylim(np.min(y),np.max(y))
		ax.set_ylim(lims)
		ax.set_xlim(0,1)
		ax.set_xlabel(i)
		fig.canvas.draw()

def press(event):
	# print('press', event.key)
	sys.stdout.flush()
	global i
	if event.key == 'right':
		if i+1<=99:
			i+=1
			load_data(i)
	elif event.key == 'left':
		if i-1>=0:
			i-=1
			load_data(i)

	elif event.key == 'd':
		load_data('fast')

	elif event.key in '1234567890':
		ax.set_title(ax.get_title()+event.key)
		fig.canvas.draw()

	elif event.key == 'enter':
		num=int(ax.get_title())
		ax.set_title('')
		if (num>=0)&(num<100):
			load_data(num)

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)

def onclick(event):
	global i
	if (event.dblclick)&(ax.get_xlabel()=='fast'):
		num=int(round(event.xdata*100))-1
		i=num
		load_data(i)
	print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
		  ('double' if event.dblclick else 'single', event.button,
		   event.x, event.y, event.xdata, event.ydata))




cid = fig.canvas.mpl_connect('button_press_event', onclick)
# fig.canvas.mpl_disconnect(cid)
line,= ax.plot([],[])
i=10
load_data(i)

plt.gcf().text(0.02, 0.5, 'd - обзорный режим\nc - предыдущий масштаб\nv-следующий масштаб\nСтрелки - переключение участков\nДвойной клик - выбор участка', fontsize=8)
# plt.grid(True)
plt.subplots_adjust(left=0.4)
plt.show()