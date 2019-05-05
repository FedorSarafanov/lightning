import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import CheckButtons 
# import scipy.fftpack
import sys


class CurrentView(object):

	def __new__(cls, path = '../numpydata/'):
		# print(path)
		return super(CurrentView, cls).__new__(cls)

	def __init__(self, path = '../numpydata/'):
		# self.channel = 2 # default channel



		self.visible_channels = [0,1,2,3]
		self.channels = [0,1,2,3]
	
		self.num = -1 # default view is fast
		self.fig, self.ax = plt.subplots()
		self.ax.set_title(str(self.visible_channels))
		plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
		self.toolbar = self.fig.canvas.manager.toolbar
		# self.line, = self.ax.plot([], [])

		self.rax = plt.axes([0.05, 0.4, 0.1, 0.15])
		self.check = CheckButtons(self.rax, self.channels, [1,1,1,1])
		self.check.on_clicked(self.func)

		self.lines=[self.ax.plot([], [])[0],self.ax.plot([], [])[0],self.ax.plot([], [])[0],self.ax.plot([], [])[0]]

		# ch = np.load('../numpydata/fastview.npy')[self.channel]
		self.fast = np.load('../numpydata/fastview.npy')

		self.limits = {channel:(np.max(self.fast[channel]), np.min(self.fast[channel])) for channel in self.channels}
		self.lims = (np.min([np.min(self.fast[ch]) for ch in self.visible_channels]), np.max([np.max(self.fast[ch]) for ch in self.visible_channels]))
		# print(self.limits)

		for ch in self.visible_channels:
			self.lines[ch].set_data(np.linspace(0, 1, len(self.fast[ch])), self.fast[ch])
			self.ax.set_xlim(0, 1)
			# self.ax.set_ylim(self.limits[ch])
			self.ax.set_ylim(self.lims)

		self.ax.set_xlabel('-1')
		self.btn = self.fig.canvas.mpl_connect('key_press_event', self.on_press)
		self.clk = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
		self.fig.canvas.draw()

	def show(self):
		plt.show()

	def on_modify_visible_channels(self):
		pass
		self.ax.set_title(str(self.visible_channels))
		if self.num==-1:
			self.lims = (np.min([np.min(self.fast[ch]) for ch in self.visible_channels]), np.max([np.max(self.fast[ch]) for ch in self.visible_channels]))

			for ch in self.channels:
				if ch not in self.visible_channels:
					self.lines[ch].set_visible(False)
				else:
					self.lines[ch].set_visible(True)


			# for ch in self.visible_channels:
				# self.lines[ch].set_visible(not self.lines[ch].get_visible())
				# self.lines[ch].set_data(np.linspace(0, 1, len(self.fast[ch])), self.fast[ch])
				# self.ax.set_xlim(0, 1)
				# self.ax.set_ylim(self.limits[ch])
				# self.ax.set_ylim(self.lims)

			self.ax.set_ylim(self.lims)
			self.fig.canvas.draw()

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

		# elif event.key in '0123':
			# pass
			# tempnum = int(event.key)
			# if tempnum in self.visible_channels:
			# 	if len(self.visible_channels)>=2:
			# 		self.visible_channels.remove(tempnum)
			# else:
			# 	self.visible_channels.append(tempnum)

			# self.visible_channels=sorted(self.visible_channels)
			# self.on_modify_visible_channels()
			# self.ax.set_title(self.ax.get_title()+event.key)
			# self.fig.canvas.draw()

		# elif event.key == 'enter':
		# 	try:
		# 		tempnum = int(self.ax.get_title())
		# 	except:
		# 		pass
		# 	else:
		# 		self.ax.set_title('')
		# 		if (tempnum >= 0) & (tempnum < 100):
		# 			self.num = tempnum
		# 		# self.load_data()
		else:
			pass


	def on_click(self, event):
		if event.inaxes!=self.rax:
			if (event.dblclick)&(self.num ==-1):
				self.num = int(round(event.xdata*100))-1
				self.load_data()

			print('%s click: button = %d, x = %d, y = %d, xdata = %f, ydata = %f' %
				 ('double' if event.dblclick else 'single', event.button,
				 event.x, event.y, event.xdata, event.ydata))

	def func(self,label):
		# index=int(label)
		bools = self.check.get_status()
		# plt.draw()
		# tempnum = index
		# if tempnum in self.visible_channels:
		# 	if len(self.visible_channels)>=2:
		# 		self.visible_channels.remove(tempnum)
		# else:
		# 	self.visible_channels.append(tempnum)
		indexs=[]
		for ch in self.channels:
			if bools[ch]==True:
				indexs.append(ch)
		self.visible_channels=np.array(self.channels)[indexs]
		self.on_modify_visible_channels()
				# self.lines[ch].set_visible(False)
			# else:
				# self.lines[ch].set_visible(True)


	def load_data(self):

		if self.ax.get_xlabel() != str(self.num):
			if self.num == -1:
				for ch in self.visible_channels:
					self.lines[ch].set_data(np.linspace(0, 1, len(self.fast[ch])), self.fast[ch])
					self.ax.set_xlim(0, 1)
					# self.ax.set_ylim(self.limits[ch])
					self.ax.set_ylim(self.lims)
			else:
				data = np.load('../numpydata/data'+str(self.num)+'.npy')
				for ch in self.visible_channels:
					self.lines[ch].set_data(np.linspace(self.num/100, (self.num+1)/100, len(data[ch])), data[ch])
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