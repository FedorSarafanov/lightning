import numpy as np
from matplotlib import pyplot as plt 
import scipy.fftpack
import sys

i=10
data=np.load('../numpydata/data'+str(i)+'.npy')
y=data[2]

def press(event):
	print('press', event.key)
	sys.stdout.flush()
	global i
	if event.key == 'right':
		if i+1<=99:
			i+=1
			data=np.load('../numpydata/data'+str(i)+'.npy')
			y=data[2]
			line.set_xdata(np.linspace(0,1,len(y)))
			line.set_ydata(y)
			ax.set_ylim(np.min(y),np.max(y))
			ax.set_xlabel(str(i))
			fig.canvas.draw()
	elif event.key == 'left':
		if i-1>=0:
			i-=1
			data=np.load('../numpydata/data'+str(i)+'.npy')
			y=data[2]
			line.set_xdata(np.linspace(0,1,len(y)))
			line.set_ydata(y)
			ax.set_ylim(np.min(y),np.max(y))
			ax.set_xlabel(str(i))
			fig.canvas.draw()
	elif event.key == 'd':
		data=np.load('../numpydata/fastview.npy')
		y=data[2]
		print(y)
		line.set_xdata(np.linspace(0,1,len(y)))
		line.set_ydata(y)
		ax.set_ylim(np.min(y),np.max(y))
		ax.set_xlabel('global view')
		fig.canvas.draw()

	elif event.key in ['1','2','3','4','5','6','7','8','9','0']:
		ax.set_title(ax.get_title()+event.key)
		fig.canvas.draw()
	elif event.key == 'enter':
		num=int(ax.get_title())
		ax.set_title('')
		if (num>=0)&(num<100):
			i=num
			data=np.load('../numpydata/data'+str(i)+'.npy')
			y=data[2]
			line.set_xdata(np.linspace(0,1,len(y)))
			line.set_ydata(y)
			ax.set_ylim(np.min(y),np.max(y))
			ax.set_xlabel(str(i))
			fig.canvas.draw_idle()



fig, ax = plt.subplots()

fig.canvas.mpl_connect('key_press_event', press)
line,= ax.plot(np.linspace(0,1,len(y)),y)
ax.set_xlabel(str(i))
plt.show()