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

data = np.loadtxt('../../data/RXJ1131_ALL.rdb',skiprows=2)

def size_estimate(i,d): # i is , d is either data or dataa2
    '''
    inferring the size of the error relative to the magnitude in the real data
    :param i: columns of magnitudes, for this dataset, it is either 1,3,5,7
    :param d: dataset, ie: data
    :return: mean of the fraction sqrt(f1^2 + f2^3+ ...)/mean(error in mag)
    '''
    F = d[:,i]
    F_error = np.mean(d[:,i+1])
    f2 = []
    for i in range(len(f)):
        f2.append(f[i]**2)
    return np.int((np.sqrt(np.sum(np.array(f2))))/F_error)
a,b,c,d = size_estimate(1,data),size_estimate(3,data),size_estimate(5,data),size_estimate(7,data)
#print(a,b,c,d)

# calculating the a th, b th, c th, d th of the power of the signal
def std_gausian(fraction,mag):
    f2 = []
    for i in range(len(mag)):
        f2.append(mag[i]**2)
    return (np.sqrt(np.sum(np.array(f2))))/fraction
error0,error1,error2,error3 = std_gausian(a,fs),std_gausian(b,fs),std_gausian(c,fs),std_gausian(d,fs)
print(error0,error1,error2,error3)

# generating noise
noise0= np.random.normal(0,error0,size = len(fs[0]))
noise1= np.random.normal(0,error1,size = len(fs[1]))
noise2= np.random.normal(0,error2,size = len(fs[2]))
noise3= np.random.normal(0,error3,size = len(fs[3]))
noise = np.array([noise0,noise1,noise2,noise3])

for i in range(4):
    plt.plot(ts[i], fs[i]+noise[i], color='black', label = 'with noise')
    plt.plot(ts[i], fs[i], color='orange', label='without noise')
    plt.legend()
    plt.xlabel('time[days]')
    plt.ylabel('magnitude')
    plt.show()

