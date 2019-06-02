from numpy import *

t = array([2,	2.03,2.05,2.09,2.2,2.21,2.22])
H    = array([10,	2, 2,10,10,10,10])

dt = 0.06

def re_strikes_count(time, field, field_min, delta_t=0.06):
	time = time[field>field_min]
	count = 1
	for i in range(0,len(field)-1):
		if time[i+1]-time[i]<delta_t:
			count+=1
		else:
			break
	return count

print(re_strikes_count(t,H,5))
# delta = time[1:]-time[:-1]
# # print(delta)
# I = indices((len(delta),))[0]
# T = I[delta<dt]
# dd = T[1:]-T[:-1]
# # print(dd)
# dd[dd>1] = 0 
# # print(dd)
# temp = split(dd,where(dd==0)[0])
# # if len(temp[0])
# print(temp)
# # L = len(split(dd,where(dd==0)[0])[0])
# # print(L)