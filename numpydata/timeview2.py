import numpy as np
from matplotlib import pyplot as plt 
# import scipy.fftpack
import sys


class CurrentView(object):

	def __new__(cls, path = '../numpydata/'):
		# print(path)
		return super(CurrentView, cls).__new__(cls)

	def __init__(self, path = '../numpydata/'):
		self.channel = 2 # default channelzz
		self.num = -1 # default view is fast
		self.fig, self.ax = plt.subplots()
		self.toolbar = self.fig.canvas.manager.toolbar
		self.line, = self.ax.plot([], [])

		ch = np.load('../numpydata/fastview.npy')[self.channel]
		self.lims = (np.min(ch), np.max(ch))

		self.line.set_data(np.linspace(0, 1, len(ch)), ch)
		self.ax.set_xlim(0, 1)
		self.ax.set_ylim(self.lims)
		self.ax.set_xlabel('-1')
		self.btn = self.fig.canvas.mpl_connect('key_press_event', self.on_press)
		self.clk = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
		self.fig.canvas.draw()

	def show(self):
		plt.show()

	def on_press(self, event):
	# print('press', event.key)
		sys.stdout.flush()
		if event.key == 'right':
			if self.num+1 <= 99:
				self.num += 1
				self.load_data()
		elif event.key == 'left':
			if self.num-1 >= 0:
				self.num -= 1
				self.load_data()

		elif event.key == 'd':
			self.num = -1
			self.load_data()

		elif event.key in '1234567890':
			self.ax.set_title(self.ax.get_title()+event.key)
			self.fig.canvas.draw()

		elif event.key == 'enter':
			try:
				tempnum = int(self.ax.get_title())
			except:
				pass
			else:
				self.ax.set_title('')
				if (tempnum >= 0) & (tempnum < 100):
					self.num = tempnum
				self.load_data()
		else:
			pass


	def on_click(self, event):
		if (event.dblclick)&(self.num ==-1):
			self.num = int(round(event.xdata*100))-1
			self.load_data()

		print('%s click: button = %d, x = %d, y = %d, xdata = %f, ydata = %f' %
			 ('double' if event.dblclick else 'single', event.button,
			 event.x, event.y, event.xdata, event.ydata))

	def load_data(self):
		if self.ax.get_xlabel() != str(self.num):
			if self.num == -1:
				data = np.load('../numpydata/fastview.npy')
				self.line.set_data(np.linspace(0, 1, len(data[self.channel])), data[self.channel])
				self.ax.set_xlim(0, 1)
			else:
				data = np.load('../numpydata/data'+str(self.num)+'.npy')
				self.line.set_data(np.linspace(self.num/100, (self.num+1)/100, len(data[self.channel])), data[self.channel])
				self.ax.set_xlim(self.num/100, (self.num+1)/100)

			self.ax.set_ylim(self.lims)
			self.ax.set_xlabel(self.num)
			self.fig.canvas.draw()
		if (self.ax.get_xlabel() == str(self.num))&(str(self.num) == '-1'):
			self.ax.set_xlim(0, 1)
			self.ax.set_ylim(self.lims)
			self.fig.canvas.draw()

obj = CurrentView()
obj.show()