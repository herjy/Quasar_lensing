from simulation import index, simulation_, plot
import numpy as np

t_drive_new = np.loadtxt('t_drive_new.txt')


ind = index(15,t_drive_new) # 15 means that the time delay is chosen randomly between 0 and 15 days

index0 = ind[0]
index1 = ind[1]
index2 = ind[2]
index3 = ind[3]
index_truth = ind[4]
time_delay = ind[5]
sample = ind[5]
np.savetxt('sample.txt',sample)

#running simulation
f = simulation_(sample)

# finding the magnitudes corresponding to each time shifted sample

f_t0 = f[index0]
f_t1 = f[index1]
f_t2 = f[index2]
f_t3 = f[index3]
f_truth = f[index_truth]

# plotting time shifted curves, and a plot where they are all alligned.

plot(f_t0,f_t1,f_t2,f_t3, time_delay,t_drive_new)
