from simulation_interpolation.simulation_functions import index, simulation_, plot
from simulation_interpolation.interpolation_functions import T_K
import numpy as np
import matplotlib.pyplot as plt

t_drive_new = np.loadtxt('../t_drive_new.txt') # true time sample.
ind = index(15,t_drive_new) # 15 means that the time delay is chosen randomly between 0 and 15 days

# getting the indices of each time shifter curve
index0 = ind[0]
index1 = ind[1]
index2 = ind[2]
index3 = ind[3]
sample = np.sort(ind[5])

#getting the time sample of each curve
ts_0 = sample[index0]
ts_1 = sample[index1]
ts_2 = sample[index2]
ts_3 = sample[index3]
ts = np.array([ts_0,ts_1,ts_2,ts_3])

# finding the magnitudes corresponding to each time shifted sample
f = simulation_(sample) #running simulation
fs_0 = f[index0]
fs_1 = f[index1]
fs_2 = f[index2]
fs_3 = f[index3]

# desired sampling for each curve
tk_0 = T_K(3,ts_0)[0]
tk_1 = T_K(3,ts_1)[0]
tk_2 = T_K(3,ts_2)[0]
tk_3 = T_K(3,ts_3)[0]
tk = np.array([tk_0,tk_1,tk_2,tk_3])

h=3
time_delays = ind[6]

# sinc matrix  # make this into for loop
a = np.array([(ts_0[:,np.newaxis]+time_delays[0])-tk_0[np.newaxis,:]])
b = np.array([(ts_1[:,np.newaxis]+time_delays[1])-tk_1[np.newaxis,:]])
c = np.array([(ts_2[:,np.newaxis]+time_delays[2])-tk_2[np.newaxis,:]])
d = np.array([(ts_3[:,np.newaxis]+time_delays[3])-tk_3[np.newaxis,:]])
sinc_matrix = np.sinc(np.concatenate([a,b,c,d],axis =0)/h)
#sinc_matrix.shape # 4 x s x k

B= np.concatenate([fs_0[np.newaxis,:],fs_1[np.newaxis,:],fs_2[np.newaxis,:],fs_3[np.newaxis,:]])

A = sinc_matrix
Y = B

mu = 0.0038  # just to not get stuck

X = np.zeros(len(tk_0))

count = 0
R = [np.sum(Y ** 2)]
epsilon = 0.3


while (R[-1] > epsilon) and (
        count < 5000):  # calculated this norm^2 for the least squares sol it gave 0.5 so I chose 0.5>0.3
    X_new = X + mu * ((A[0].T) @ (Y[0] - A[0] @ X))
    X = X_new.copy()
    X_new = X + mu * ((A[1].T) @ (Y[1] - A[1] @ X))
    X = X_new.copy()
    X_new = X + mu * ((A[2].T) @ (Y[2] - A[2] @ X))
    X = X_new.copy()
    X_new = X + mu * ((A[3].T) @ (Y[3] - A[3] @ X))
    X = X_new.copy()
    R.append(np.sum((Y - A @ X) ** 2.))
    count += 1



plt.title('Convergence')
plt.plot(np.array(R[1:]))
plt.xlabel('iterations')
plt.ylabel('error')
plt.show()

cut = np.logical_and(X < np.max(Y), X > np.min(Y))
plt.plot(ts_0, Y[0], 'o', label='light curve 1')
plt.plot(ts_1, Y[1], 'o', label='light curve 2')
plt.plot(ts_2, Y[2], 'o', label='light curve 3')
plt.plot(ts_3, Y[3], 'o', label='light curve 4')

plt.plot(tk[0][cut], X[cut],'o', label='interpolated', color='black')
plt.xlabel('time[days]')
plt.ylabel('magnitude')
plt.title('sinc interpolated values')
plt.legend()
plt.show()

#print(B[1].shape)