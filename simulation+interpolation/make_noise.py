from simulation_functions import index, simulation_, plot
from interpolation_functions import T_K
import numpy as np
import matplotlib.pyplot as plt

t_drive_new = np.loadtxt('t_drive_new.txt') # true time sample.
ind = index(15,t_drive_new) # 15 means that the time delay is chosen randomly between 0 and 15 days

# getting the indices of each time shifter curve
index0, index1,index2,index3 = ind[0],ind[1],ind[2],ind[3]
sample = np.sort(ind[5])

#getting the time sample of each curve
ts = np.array([sample[index0],sample[index1],sample[index2],sample[index3]])

# finding the magnitudes corresponding to each time shifted sample
f = simulation_(sample) #running simulation
fs = np.array([f[index0],f[index1],f[index2],f[index3]])


# desired sampling for each curve
# tk = np.array([T_K(3,ts[0])[0],T_K(3,ts[1])[0],T_K(3,ts[2])[0],T_K(3,ts[3])[0]])
h=3
time_delays = ind[6]

# generating noise
noise0= np.random.normal(0,10,size = len(ts[0]))

plt.plot(ts[0]+noise0, fs[0], color='black', label = 'without noise')
plt.plot(ts[0], fs[0], color='orange', label='without noise')
plt.legend()
plt.xlabel('time[days]')
plt.ylabel('magnitude')
# plt.legend()
plt.show()