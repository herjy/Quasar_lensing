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
tk = np.array([T_K(3,ts[0])[0],T_K(3,ts[1])[0],T_K(3,ts[2])[0],T_K(3,ts[3])[0]])

h=3
time_delays = ind[6]

# sinc matrix
matrix = []
for i in range(4):
    matrix.append(np.array([(ts[i][:,np.newaxis]+time_delays[i])-tk[i][np.newaxis,:]]))
matrix = np.array(matrix)

A = np.sinc(np.concatenate([matrix[0],matrix[1],matrix[2],matrix[3]],axis =0)/h)

Y= np.concatenate([fs[0][np.newaxis,:],fs[1][np.newaxis,:],fs[2][np.newaxis,:],fs[3][np.newaxis,:]])

def AT(X,A,Y): # X = Y-A@X
    gradient =[]
    for i in range(4):
        gradient.append(np.dot(A[i].T, Y[i] - A[i] @ X))
    gradient = np.array(gradient)
    return np.mean(gradient)

X = np.zeros(len(tk[0]))

#
# def linorm(S, nit):
#     n1, n2 = np.shape(S)
#     x0 = np.random.rand(1, n1)
#     x0 = x0 / np.sqrt(np.sum(x0 ** 2))
#     for i in range(nit):
#         x = np.dot(x0, S)
#         xn = np.sqrt(np.sum(x ** 2))
#         xp = x / xn
#         y = np.dot(xp, S.T)
#         yn = np.sqrt(np.sum(y ** 2))
#         if yn < np.dot(y, x0.T):
#             break
#         x0 = y / yn
#     return 1. / xn
#
mu = 0.0038  # just to not get stuck

X = np.zeros(len(tk[0]))

count = 0
R = [np.sum(Y ** 2)]
epsilon = 0.3



while (R[-1] > epsilon) and (
        count < 5000):  # calculated this norm^2 for the least squares sol it gave 0.5 so I chose 0.5>0.3
    X_new = X + mu * AT(X,A,Y)
    X = X_new.copy()
    R.append(np.sum((Y - A @ X) ** 2.))
    count += 1


plt.title('Convergence')
plt.plot(np.array(R[1:]))
plt.xlabel('iterations')
plt.ylabel('error')
plt.show()

cut = np.logical_and(X < np.max(Y), X > np.min(Y))
plt.plot(ts[0], Y[0], 'o', label='light curve 1')
plt.plot(ts[1], Y[1], 'o', label='light curve 2')
plt.plot(ts[2], Y[2], 'o', label='light curve 3')
plt.plot(ts[3], Y[3], 'o', label='light curve 4')

plt.plot(tk[0][cut], X[cut], label='interpolated', color='black')
plt.xlabel('time[days]')
plt.ylabel('magnitude')
plt.title('sinc interpolated values')
plt.legend()
plt.show()

