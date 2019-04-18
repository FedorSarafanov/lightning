# This program reading data from dsf files. Writen by 
# Fedor Sarafanov (sfg180@yandex.ru), program based on matlab 
# function, writen by Fedor Kuterin (xredor@gmail.com), 28.03.2019.

# For read, maybe needs modify "seek(0x4a)"  to "seek (0x...)", where
# 0x... is address start data blocks. It's maybe found with HEX-editor.

import numpy as np
from matplotlib import pyplot as plt 
import scipy.fftpack

types={
	0:np.ubyte, 1:np.byte, 2:np.uint16, 3:np.int16, 4:np.uint16,
	5:np.uint16, 6:np.float32, 7:np.float64, 8:np.longdouble
}


def dsfread(filename):
	with open(filename, "rb") as f:

		f.seek(0,0)
		data_type = int.from_bytes(f.read(1), byteorder='little')

		f.seek(9,0)
		channel_count = int.from_bytes(f.read(2), byteorder='little')

		f.seek(0x4a,0)
		data = np.fromfile(f, dtype=types[data_type]).reshape(-1,\
			channel_count).T

	return data

print('Data reading, please wait...')
data=dsfread("data.dsf")
def k(i):
	if i==0:
		return i
	else:
		return 1

W=len(data[0])/100
for i in range(0,100):
	start=int(i*W+k(i))
	final=int((i+1)*W)
	temp=data.T[start:final].T 
	np.save('../numpydata/data'+str(i)+'.npy', temp)
	print('writen ',i)

print('Data writen to numpy binary files')

a0=data[0][::10000]
a1=data[1][::10000]
a2=data[2][::10000]
a3=data[3][::10000]

temp=np.array([a0,a1,a2,a3])
np.save('../numpydata/fastview.npy', temp)

print('Fast data writen to numpy binary files')
