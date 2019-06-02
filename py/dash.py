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
rc('text', usetex=True)
rc('text.latex',preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex',preamble=r'\usepackage{cmap}')
rc('text.latex',preamble=r'\usepackage[T2A]{fontenc}')
rc('text.latex',preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex', preamble='\\usepackage[english,russian]{babel}')
rc('text.latex', preamble='\\usepackage{amsmath}')
plt.rcParams.update({'font.size': 14})
plt.xlabel(r'$\xi = \frac{\omega}{c}$')


# plt.yticks([])
plt.plot(np.linspace(0,10,100),np.sin(np.linspace(0,10,100)))
plt.show()

# plt.plot(obj.raw_time, dy)

# obj.plot()
# obj.show()